"""
Sistema Militar de Reconhecimento com Drones - Versão Aprimorada
Implementação das correções críticas identificadas na avaliação
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import threading
import queue
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

class ThreatLevel(Enum):
    BAIXO = "baixo"
    MEDIO = "medio"
    ALTO = "alto"
    CRITICO = "critico"

class WeatherCondition(Enum):
    CLARO = "claro"
    NUBLADO = "nublado"
    CHUVA_LEVE = "chuva_leve"
    NEBLINA = "neblina"
    NOTURNO = "noturno"

@dataclass
class Detection:
    id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float
    gps_coords: Tuple[float, float]
    altitude: float
    threat_level: ThreatLevel
    thermal_signature: Optional[float] = None
    movement_vector: Optional[Tuple[float, float]] = None
    track_id: Optional[int] = None

@dataclass
class SystemMetrics:
    detection_rate: float
    false_positive_rate: float
    processing_latency: float
    system_load: Dict[str, float]
    thermal_status: float
    mission_time_remaining: float

class MultiModalDetector:
    """
    Detector multi-modal com capacidades RGB, térmica e baixa luminosidade
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # Modelos especializados
        self.rgb_model = self._load_model("models/yolov8x_military.pt")
        self.thermal_model = self._load_model("models/thermal_detector.pt")
        self.lowlight_model = self._load_model("models/lowlight_enhanced.pt")
        
        # Sistema de fusão de sensores
        self.fusion_weights = {
            'rgb': 0.4,
            'thermal': 0.4,
            'lowlight': 0.2
        }
        
        # Tracking avançado
        self.tracker = cv2.TrackerCSRT_create()
        self.active_tracks = {}
        self.next_track_id = 1
        
        # Histórico para validação temporal
        self.detection_history = {}
        
        # Sistema de camuflagem
        self.camouflage_detector = self._init_camouflage_detector()
        
        # Métricas em tempo real
        self.metrics = SystemMetrics(0, 0, 0, {}, 0, 0)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração do sistema"""
        default_config = {
            'detection_threshold': 0.75,  # Aumentado de 0.55
            'nms_threshold': 0.4,
            'min_detection_size': 20,
            'max_altitude': 300,  # Aumentado de 150m
            'temporal_consistency_frames': 5,  # Aumentado de 4
            'tracking_max_age': 30,
            'fusion_confidence_boost': 0.15,
            'thermal_weight_night': 0.7,
            'camouflage_sensitivity': 0.8
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            default_config.update(config)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
        
        return default_config
    
    def _load_model(self, model_path: str):
        """Carrega modelo especializado"""
        try:
            if "thermal" in model_path:
                # Modelo especializado para infravermelho
                model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)
                model.conf = 0.7  # Threshold mais alto para térmica
            elif "lowlight" in model_path:
                # Modelo com enhancement para baixa luminosidade
                model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)
                model.conf = 0.65
            else:
                # Modelo RGB principal - YOLOv8x para melhor precisão
                model = torch.hub.load('ultralytics/yolov8', 'yolov8x')
                model.conf = self.config['detection_threshold']
            
            # Otimização para hardware embarcado
            if torch.cuda.is_available():
                model.to('cuda')
                model.half()  # FP16 para speed
            
            return model
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            return None
    
    def _init_camouflage_detector(self):
        """Inicializa detector de camuflagem usando análise de textura"""
        # Filtros Gabor para detecção de padrões de camuflagem
        kernels = []
        for theta in range(0, 180, 30):
            for frequency in [0.1, 0.3, 0.5]:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                         2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
        return kernels
    
    def detect_camouflage(self, image_region: np.ndarray) -> float:
        """Detecta padrões de camuflagem em região da imagem"""
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        responses = []
        
        for kernel in self.camouflage_detector:
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            responses.append(np.var(response))
        
        # Análise estatística das respostas
        response_variance = np.var(responses)
        response_mean = np.mean(responses)
        
        # Padrões de camuflagem têm alta variância em múltiplas orientações
        camouflage_score = response_variance / (response_mean + 1e-6)
        return min(camouflage_score / 1000.0, 1.0)  # Normalizar para [0,1]
    
    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """Melhora imagens em condições de baixa luminosidade"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Redução de ruído
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def multi_modal_detection(self, rgb_frame: np.ndarray, 
                            thermal_frame: Optional[np.ndarray] = None,
                            altitude: float = 100.0,
                            weather: WeatherCondition = WeatherCondition.CLARO) -> List[Detection]:
        """
        Detecção multi-modal com fusão de sensores
        """
        start_time = time.time()
        detections = []
        
        # Ajuste de pesos baseado em condições
        weights = self.fusion_weights.copy()
        if weather == WeatherCondition.NOTURNO:
            weights['thermal'] = 0.7
            weights['rgb'] = 0.2
            weights['lowlight'] = 0.1
        elif weather in [WeatherCondition.NEBLINA, WeatherCondition.CHUVA_LEVE]:
            weights['thermal'] = 0.6
            weights['rgb'] = 0.3
            weights['lowlight'] = 0.1
        
        # Processamento paralelo dos modelos
        with ThreadPoolExecutor(max_workers=3) as executor:
            # RGB Detection
            rgb_future = executor.submit(self._detect_rgb, rgb_frame, altitude)
            
            # Thermal Detection (se disponível)
            thermal_future = None
            if thermal_frame is not None:
                thermal_future = executor.submit(self._detect_thermal, thermal_frame)
            
            # Low-light enhanced detection
            enhanced_frame = self.enhance_low_light(rgb_frame)
            lowlight_future = executor.submit(self._detect_lowlight, enhanced_frame, altitude)
            
            # Coleta de resultados
            rgb_detections = rgb_future.result()
            thermal_detections = thermal_future.result() if thermal_future else []
            lowlight_detections = lowlight_future.result()
        
        # Fusão das detecções
        fused_detections = self._fuse_detections(
            rgb_detections, thermal_detections, lowlight_detections, weights
        )
        
        # Validação temporal e tracking
        validated_detections = self._temporal_validation(fused_detections)
        
        # Análise de camuflagem para cada detecção
        for detection in validated_detections:
            x, y, w, h = detection.bbox
            roi = rgb_frame[y:y+h, x:x+w]
            if roi.size > 0:
                camouflage_score = self.detect_camouflage(roi)
                if camouflage_score > self.config['camouflage_sensitivity']:
                    detection.confidence += 0.1  # Boost para possível camuflagem
                    detection.threat_level = ThreatLevel.ALTO
        
        # Atualização de métricas
        processing_time = time.time() - start_time
        self.metrics.processing_latency = processing_time * 1000  # ms
        
        return validated_detections
    
    def _detect_rgb(self, frame: np.ndarray, altitude: float) -> List[Dict]:
        """Detecção RGB padrão"""
        if self.rgb_model is None:
            return []
        
        results = self.rgb_model(frame)
        detections = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if box.cls == 0:  # Classe 'person'
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Ajuste de confiança baseado na altitude
                        altitude_factor = max(0.5, 1.0 - (altitude - 100) / 300)
                        adjusted_conf = conf * altitude_factor
                        
                        if adjusted_conf > self.config['detection_threshold']:
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                'confidence': float(adjusted_conf),
                                'source': 'rgb'
                            })
        
        return detections
    
    def _detect_thermal(self, frame: np.ndarray) -> List[Dict]:
        """Detecção térmica especializada"""
        if self.thermal_model is None:
            return []
        
        results = self.thermal_model(frame)
        detections = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    if conf > 0.7:  # Threshold mais alto para térmica
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': float(conf),
                            'source': 'thermal'
                        })
        
        return detections
    
    def _detect_lowlight(self, frame: np.ndarray, altitude: float) -> List[Dict]:
        """Detecção em baixa luminosidade"""
        if self.lowlight_model is None:
            return []
        
        results = self.lowlight_model(frame)
        detections = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    if conf > 0.65:
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': float(conf),
                            'source': 'lowlight'
                        })
        
        return detections
    
    def _fuse_detections(self, rgb_dets: List[Dict], thermal_dets: List[Dict], 
                        lowlight_dets: List[Dict], weights: Dict[str, float]) -> List[Detection]:
        """Fusão inteligente de detecções multi-modais"""
        all_detections = []
        
        # Combinar todas as detecções
        for det in rgb_dets:
            det['weight'] = weights['rgb']
            all_detections.append(det)
        
        for det in thermal_dets:
            det['weight'] = weights['thermal']
            all_detections.append(det)
        
        for det in lowlight_dets:
            det['weight'] = weights['lowlight']
            all_detections.append(det)
        
        if not all_detections:
            return []
        
        # Clustering de detecções próximas
        fused_detections = []
        used = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used:
                continue
                
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(all_detections[i+1:], i+1):
                if j in used:
                    continue
                
                # Calcular IoU
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > 0.3:  # Detecções próximas
                    cluster.append(det2)
                    used.add(j)
            
            # Fusão do cluster
            if len(cluster) > 1:
                fused = self._merge_cluster(cluster)
            else:
                fused = cluster[0]
            
            # Converter para objeto Detection
            detection = Detection(
                id=len(fused_detections),
                confidence=fused['confidence'],
                bbox=fused['bbox'],
                timestamp=time.time(),
                gps_coords=(0.0, 0.0),  # Será preenchido pelo sistema de navegação
                altitude=100.0,  # Será preenchido pelo sistema de navegação
                threat_level=self._assess_threat_level(fused['confidence'])
            )
            
            fused_detections.append(detection)
        
        return fused_detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Calcula Intersection over Union entre duas bounding boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge um cluster de detecções"""
        # Média ponderada das confidences
        total_weight = sum(det['weight'] * det['confidence'] for det in cluster)
        total_weight_sum = sum(det['weight'] for det in cluster)
        avg_confidence = total_weight / total_weight_sum if total_weight_sum > 0 else 0
        
        # Média das bounding boxes
        boxes = [det['bbox'] for det in cluster]
        avg_x = sum(box[0] for box in boxes) // len(boxes)
        avg_y = sum(box[1] for box in boxes) // len(boxes)
        avg_w = sum(box[2] for box in boxes) // len(boxes)
        avg_h = sum(box[3] for box in boxes) // len(boxes)
        
        return {
            'bbox': (avg_x, avg_y, avg_w, avg_h),
            'confidence': min(avg_confidence + 0.1, 1.0),  # Boost por consenso
            'sources': [det['source'] for det in cluster]
        }
    
    def _temporal_validation(self, detections: List[Detection]) -> List[Detection]:
        """Validação temporal para reduzir falsos positivos"""
        validated = []
        current_time = time.time()
        
        for detection in detections:
            # Buscar detecções similares no histórico
            similar_count = 0
            for hist_time, hist_detections in self.detection_history.items():
                if current_time - hist_time < 2.0:  # Últimos 2 segundos
                    for hist_det in hist_detections:
                        iou = self._calculate_iou(detection.bbox, hist_det.bbox)
                        if iou > 0.5:
                            similar_count += 1
            
            # Validar se tem consistência temporal
            if similar_count >= self.config['temporal_consistency_frames'] - 1:
                detection.confidence = min(detection.confidence + 0.05, 1.0)
                validated.append(detection)
            elif detection.confidence > 0.85:  # Alta confiança passa direto
                validated.append(detection)
        
        # Atualizar histórico
        self.detection_history[current_time] = detections
        
        # Limpar histórico antigo
        cutoff_time = current_time - 5.0
        self.detection_history = {
            t: dets for t, dets in self.detection_history.items() 
            if t > cutoff_time
        }
        
        return validated
    
    def _assess_threat_level(self, confidence: float) -> ThreatLevel:
        """Avalia nível de ameaça baseado na confiança"""
        if confidence > 0.9:
            return ThreatLevel.CRITICO
        elif confidence > 0.8:
            return ThreatLevel.ALTO
        elif confidence > 0.7:
            return ThreatLevel.MEDIO
        else:
            return ThreatLevel.BAIXO


class MilitaryDroneSystem:
    """
    Sistema principal integrado para operações militares
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.detector = MultiModalDetector(config_path)
        self.db_path = "military_recon.db"
        self.init_database()
        
        # Sistema de comunicação redundante
        self.comm_systems = ['wifi', 'radio', 'satellite']
        self.active_comm = 'wifi'
        
        # Monitoramento de sistema
        self.system_monitor = SystemMonitor()
        
        # Thread para processamento contínuo
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
    def init_database(self):
        """Inicializa banco de dados para logs operacionais"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                confidence REAL,
                bbox TEXT,
                gps_coords TEXT,
                altitude REAL,
                threat_level TEXT,
                mission_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def start_mission(self, mission_id: str):
        """Inicia missão de reconhecimento"""
        self.mission_id = mission_id
        self.stop_processing.clear()
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            args=(mission_id,)
        )
        self.processing_thread.start()
        
        logging.info(f"Missão {mission_id} iniciada")
    
    def stop_mission(self):
        """Para missão atual"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join()
        
        logging.info("Missão finalizada")
    
    def _processing_loop(self, mission_id: str):
        """Loop principal de processamento"""
        while not self.stop_processing.is_set():
            try:
                # Simular captura de frames (substituir por interface real do drone)
                rgb_frame = self._capture_rgb_frame()
                thermal_frame = self._capture_thermal_frame()
                
                # Obter dados de navegação
                altitude = self._get_altitude()
                gps_coords = self._get_gps_coordinates()
                weather = self._assess_weather()
                
                # Processamento
                detections = self.detector.multi_modal_detection(
                    rgb_frame, thermal_frame, altitude, weather
                )
                
                # Salvar detecções
                for detection in detections:
                    detection.gps_coords = gps_coords
                    detection.altitude = altitude
                    self._save_detection(detection, mission_id)
                
                # Enviar alertas para detecções críticas
                critical_detections = [d for d in detections 
                                     if d.threat_level == ThreatLevel.CRITICO]
                
                if critical_detections:
                    self._send_critical_alert(critical_detections)
                
                # Monitoramento de sistema
                self.system_monitor.update_metrics()
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logging.error(f"Erro no loop de processamento: {e}")
                time.sleep(1)
    
    def _capture_rgb_frame(self) -> np.ndarray:
        """Captura frame RGB do drone (placeholder)"""
        # Placeholder - substituir por interface real do drone
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def _capture_thermal_frame(self) -> Optional[np.ndarray]:
        """Captura frame térmico (placeholder)"""
        # Placeholder - substituir por interface real do sensor térmico
        return np.zeros((720, 1280, 1), dtype=np.uint8)
    
    def _get_altitude(self) -> float:
        """Obtém altitude atual (placeholder)"""
        return 100.0
    
    def _get_gps_coordinates(self) -> Tuple[float, float]:
        """Obtém coordenadas GPS (placeholder)"""
        return (0.0, 0.0)
    
    def _assess_weather(self) -> WeatherCondition:
        """Avalia condições climáticas (placeholder)"""
        return WeatherCondition.CLARO
    
    def _save_detection(self, detection: Detection, mission_id: str):
        """Salva detecção no banco de dados"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO detections 
            (timestamp, confidence, bbox, gps_coords, altitude, threat_level, mission_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection.timestamp,
            detection.confidence,
            json.dumps(detection.bbox),
            json.dumps(detection.gps_coords),
            detection.altitude,
            detection.threat_level.value,
            mission_id
        ))
        conn.commit()
        conn.close()
    
    def _send_critical_alert(self, detections: List[Detection]):
        """Envia alerta crítico para comando"""
        alert_data = {
            'timestamp': time.time(),
            'detections_count': len(detections),
            'highest_confidence': max(d.confidence for d in detections),
            'gps_coords': detections[0].gps_coords,
            'mission_id': self.mission_id
        }
        
        # Tentar múltiplos sistemas de comunicação
        for comm_system in self.comm_systems:
            try:
                success = self._send_via_comm(alert_data, comm_system)
                if success:
                    logging.info(f"Alerta crítico enviado via {comm_system}")
                    break
            except Exception as e:
                logging.warning(f"Falha no envio via {comm_system}: {e}")
                continue
    
    def _send_via_comm(self, data: Dict, comm_system: str) -> bool:
        """Envia dados via sistema de comunicação específico"""
        # Placeholder - implementar interface real
        return True


class SystemMonitor:
    """Monitor de sistema para alertas e diagnósticos"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
    
    def update_metrics(self):
        """Atualiza métricas do sistema"""
        import psutil
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'temperature': self._get_temperature(),
            'mission_time': time.time() - self.start_time
        }
        
        self.metrics_history.append(metrics)
        
        # Manter apenas últimos 1000 registros
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Verificar alertas
        self._check_system_alerts(metrics)
    
    def _get_temperature(self) -> float:
        """Obtém temperatura do sistema"""
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                return next(iter(temps.values()))[0].current
        except:
            pass
        return 45.0  # Placeholder
    
    def _check_system_alerts(self, metrics: Dict):
        """Verifica condições de alerta do sistema"""
        if metrics['cpu_percent'] > 90:
            logging.warning("CPU usage critical: {}%".format(metrics['cpu_percent']))
        
        if metrics['memory_percent'] > 85:
            logging.warning("Memory usage high: {}%".format(metrics['memory_percent']))
        
        if metrics['temperature'] > 75:
            logging.warning("System temperature high: {}°C".format(metrics['temperature']))


# Exemplo de uso
if __name__ == "__main__":
    # Inicializar sistema
    drone_system = MilitaryDroneSystem("config.json")
    
    try:
        # Iniciar missão
        drone_system.start_mission("MISSION_001")
        
        # Simular operação por 60 segundos
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
    finally:
        # Finalizar missão
        drone_system.stop_mission()
        print("Sistema finalizado")