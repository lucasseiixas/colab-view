"""
Sistema Avançado de Detecção de Highlights de Vôlei
Foco: Captura completa do Rally (Início -> Fim)
Otimizado para GPU T4 e Vídeos Longos
"""

import os
import cv2
import numpy as np
import librosa
import subprocess
import json
import tempfile
import shutil
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

# Configuração de Ambiente
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device.upper()}")
except:
    device = "cpu"

# --- CONFIGURAÇÕES GLOBAIS ---
CONFIG = {
    "chunk_duration": 120,       # Processar em blocos de 2min para economizar RAM
    "overlap": 10,               # Overlap entre chunks para não cortar rallies na borda
    "sample_rate_audio": 22050,  # Amostragem de áudio
    "frame_sample_rate": 2,      # Frames por segundo para análise visual (não precisa de 30fps)
    "min_rally_duration": 4.0,   # Rally mínimo de 4s
    "max_rally_duration": 45.0,  # Rally máximo de 45s
    "pre_event_window": 8.0,     # Quanto tempo voltar ao detectar um pico (para pegar o saque)
    "post_event_window": 6.0,    # Quanto tempo estender após o pico (para pegar comemoração)
    "merge_threshold": 2.0,      # Unir highlights separados por menos de 2s
    "audio_peak_threshold": 0.65,# Threshold normalizado para picos de áudio
    "motion_threshold": 0.15,    # Threshold para movimento de câmera
}

@dataclass
class HighlightSegment:
    start: float
    end: float
    score: float
    reason: str
    details: dict

class VolleyballRallyDetector:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.duration = self._get_video_duration()
        self.temp_dir = tempfile.mkdtemp()
        self.audio_cache = None
        
        print(f"📹 Vídeo carregado: {self.duration:.2f}s")
        print(f"📂 Temp dir: {self.temp_dir}")

    def _get_video_duration(self) -> float:
        """Obtém duração exata via ffprobe"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _extract_audio_chunk(self, start: float, duration: float) -> np.ndarray:
        """Extrai chunk de áudio específico sem carregar o vídeo todo"""
        output_path = os.path.join(self.temp_dir, f"audio_{start}.wav")
        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-t', str(duration),
            '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(CONFIG["sample_rate_audio"]), '-ac', '1',
            '-loglevel', 'quiet', output_path
        ]
        subprocess.run(cmd, check=True)
        
        y, _ = librosa.load(output_path, sr=CONFIG["sample_rate_audio"])
        os.remove(output_path)
        return y

    def _extract_frames_chunk(self, start: float, duration: float) -> List[np.ndarray]:
        """
        Extrai frames estratégicos para análise visual.
        Não extrai todos os frames, apenas amostras para detecção de cena.
        """
        output_pattern = os.path.join(self.temp_dir, f"frame_%04d.jpg")
        # Extrai 2 fps para análise de movimento/cena
        fps_cmd = f"fps={CONFIG['frame_sample_rate']}"
        
        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-t', str(duration),
            '-i', self.video_path, '-vf', fps_cmd,
            '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'quiet',
            output_pattern
        ]
        subprocess.run(cmd, check=True)
        
        frames = []
        files = sorted([f for f in os.listdir(self.temp_dir) if f.startswith("frame_") and f.endswith(".jpg")])
        for f in files:
            path = os.path.join(self.temp_dir, f)
            img = cv2.imread(path)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            os.remove(path) # Limpeza imediata
            
        return frames

    def analyze_audio_events(self, start_time: float, audio_data: np.ndarray) -> List[dict]:
        """
        Detecta eventos de áudio:
        1. Picos de volume (Gritos/Impacto)
        2. Onset detection (Início de sons percussivos - apito/impacto)
        3. Mudança de espectro (Narração vs Grito)
        """
        events = []
        
        # 1. RMS Energy (Volume geral)
        hop_length = 512
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        
        # 2. Spectral Centroid (Brilho do som - gritos são mais 'brilhantes')
        spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=CONFIG["sample_rate_audio"], hop_length=hop_length)[0]
        spec_norm = (spec_cent - np.min(spec_cent)) / (np.max(spec_cent) - np.min(spec_cent) + 1e-9)
        
        # 3. Onset Strength (Detecta ataques súbitos - apito, batida na bola)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=CONFIG["sample_rate_audio"], hop_length=hop_length)
        onset_norm = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-9)
        
        # Combinação ponderada
        combined_score = (0.4 * rms_norm) + (0.3 * spec_norm) + (0.3 * onset_norm)
        
        # Encontrar picos acima do threshold
        # Usamos um threshold dinâmico baseado no percentil 85 do chunk
        threshold = np.percentile(combined_score, 85)
        threshold = max(threshold, CONFIG["audio_peak_threshold"])
        
        peaks = []
        for i in range(1, len(combined_score) - 1):
            if combined_score[i] > threshold and combined_score[i] > combined_score[i-1] and combined_score[i] > combined_score[i+1]:
                time_pos = start_time + (i * hop_length / CONFIG["sample_rate_audio"])
                peaks.append({
                    "time": time_pos,
                    "score": float(combined_score[i]),
                    "type": "audio_peak"
                })
        
        return peaks

    def analyze_visual_context(self, frames: List[np.ndarray], start_time: float) -> List[dict]:
        """
        Analisa contexto visual para identificar:
        1. Câmera lenta (Replay)
        2. Mudança brusca de cena (Corte de câmera)
        3. Estática relativa (Momento de saque/preparação)
        """
        if len(frames) < 3:
            return []
            
        events = []
        frame_diffs = []
        
        # Calcular diferença entre frames consecutivos
        for i in range(1, len(frames)):
            prev = cv2.resize(frames[i-1], (64, 36)) # Downscale para performance
            curr = cv2.resize(frames[i], (64, 36))
            diff = cv2.absdiff(prev, curr)
            score = np.mean(diff) / 255.0
            frame_diffs.append(score)
        
        frame_diffs = np.array(frame_diffs)
        
        # Detectar cortes bruscos (Mudança de câmera durante o rally)
        # Cortes de câmera geralmente indicam ação intensa
        cut_threshold = np.mean(frame_diffs) + 2 * np.std(frame_diffs)
        
        for i, diff in enumerate(frame_diffs):
            if diff > cut_threshold and diff > 0.3: # 0.3 é um corte significativo
                t_idx = i / CONFIG["frame_sample_rate"]
                events.append({
                    "time": start_time + t_idx,
                    "score": float(diff),
                    "type": "camera_cut"
                })
        
        # Detectar padrões de "Preparação" (Baixo movimento antes do pico)
        # Isso ajuda a ancorar o início do rally
        window_size = int(2 * CONFIG["frame_sample_rate"]) # 2 segundos
        for i in range(window_size, len(frame_diffs) - window_size):
            local_mean = np.mean(frame_diffs[i-window_size:i])
            future_mean = np.mean(frame_diffs[i:i+window_size])
            
            # Se estava calmo e vai agitar -> Possível início de rally (Saque)
            if local_mean < 0.05 and future_mean > 0.15:
                t_idx = i / CONFIG["frame_sample_rate"]
                events.append({
                    "time": start_time + t_idx,
                    "score": 0.7, # Score fixo para transição calma->agitado
                    "type": "rally_start_candidate"
                })
                
        return events

    def process_chunks(self) -> List[HighlightSegment]:
        """Processa o vídeo em chunks sobrepostos"""
        all_candidates = []
        
        num_chunks = int(np.ceil(self.duration / (CONFIG["chunk_duration"] - CONFIG["overlap"])))
        
        for i in range(num_chunks):
            start = i * (CONFIG["chunk_duration"] - CONFIG["overlap"])
            if start >= self.duration:
                break
                
            duration = min(CONFIG["chunk_duration"], self.duration - start)
            print(f"⏳ Processando chunk {i+1}/{num_chunks} ({start:.1f}s - {start+duration:.1f}s)...")
            
            # 1. Extrair dados brutos
            try:
                audio_data = self._extract_audio_chunk(start, duration)
                frames = self._extract_frames_chunk(start, duration)
            except Exception as e:
                print(f"❌ Erro no chunk {i}: {e}")
                continue
            
            # 2. Analisar Áudio
            audio_events = self.analyze_audio_events(start, audio_data)
            
            # 3. Analisar Visual
            visual_events = self.analyze_visual_context(frames, start)
            
            # 4. Fundir candidatos neste chunk
            # Estratégia: Para cada pico de áudio forte, buscar contexto visual
            for aud_evt in audio_events:
                if aud_evt["score"] < 0.6: # Ignorar picos fracos
                    continue
                    
                # Verificar se há suporte visual próximo (corte de câmera ou movimento)
                visual_support = 0
                for vis_evt in visual_events:
                    if abs(vis_evt["time"] - aud_evt["time"]) < 2.0: # Dentro de 2s
                        visual_support += vis_evt["score"]
                
                final_score = aud_evt["score"] + (visual_support * 0.2)
                
                all_candidates.append({
                    "center_time": aud_evt["time"],
                    "score": final_score,
                    "audio_score": aud_evt["score"],
                    "visual_score": visual_support
                })
        
        return self._refine_segments(all_candidates)

    def _refine_segments(self, candidates: List[dict]) -> List[HighlightSegment]:
        """
        Refina os candidatos para garantir captura completa do rally.
        Lógica principal: Backtracking para achar o saque e extensão para comemoração.
        """
        if not candidates:
            return []
            
        # Ordenar por score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        final_highlights = []
        used_times = []
        
        for cand in candidates:
            center = cand["center_time"]
            
            # Evitar duplicatas muito próximas
            if any(abs(center - t) < 15.0 for t in used_times):
                continue
                
            # --- LÓGICA DE BACKTRACKING PARA INÍCIO DO RALLY ---
            # O pico de áudio geralmente é o impacto ou o grito do ponto.
            # Precisamos voltar no tempo para achar o saque.
            
            start_search = max(0, center - CONFIG["pre_event_window"])
            end_search = min(self.duration, center + CONFIG["post_event_window"])
            
            # Ajuste fino: Tentar detectar o momento exato de "silêncio antes do caos"
            # Extraímos um mini-chunk de áudio ao redor do início estimado
            try:
                # Análise local para ajustar o start exato
                local_audio = self._extract_audio_chunk(start_search, 10.0) # Pegar 10s para analisar
                rms = librosa.feature.rms(y=local_audio)[0]
                rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
                
                # Procurar o último ponto de baixa energia antes do aumento brusco
                # O rally começa quando a energia sobe consistentemente após um vale
                detected_start_offset = 0
                for k in range(len(rms_norm)-1, 0, -1):
                    if rms_norm[k] < 0.3: # Achou um vale (momento de preparação/saque)
                        detected_start_offset = k * 512 / CONFIG["sample_rate_audio"]
                        break
                
                final_start = start_search + detected_start_offset
                
                # Garantir que não comece tarde demais
                if final_start > center - 2.0:
                    final_start = max(0, center - CONFIG["pre_event_window"])
                    
            except:
                final_start = start_search

            # Definição do fim (comemoração + replay inicial)
            final_end = end_search
            
            # Validar duração mínima/máxima
            duration = final_end - final_start
            if duration < CONFIG["min_rally_duration"]:
                final_end = final_start + CONFIG["min_rally_duration"]
            elif duration > CONFIG["max_rally_duration"]:
                final_end = final_start + CONFIG["max_rally_duration"]
                
            segment = HighlightSegment(
                start=final_start,
                end=final_end,
                score=cand["score"],
                reason="Rally Detectado (Áudio + Contexto)",
                details=cand
            )
            
            final_highlights.append(segment)
            used_times.append(center)
            
            if len(final_highlights) >= 20: # Limite de segurança de highlights
                break
                
        # Ordenar por tempo cronológico
        final_highlights.sort(key=lambda x: x.start)
        
        # Merge de highlights adjacentes (ex: rally longo que foi dividido)
        merged = []
        if final_highlights:
            current = final_highlights[0]
            for next_seg in final_highlights[1:]:
                if next_seg.start - current.end < CONFIG["merge_threshold"]:
                    # Merge
                    current.end = next_seg.end
                    current.score = max(current.score, next_seg.score)
                else:
                    merged.append(current)
                    current = next_seg
            merged.append(current)
            
        return merged

    def generate_clips(self, output_dir: str, highlights: List[HighlightSegment]):
        """Gera os arquivos de vídeo finais usando FFmpeg"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n✂️ Gerando {len(highlights)} clipes...")
        
        manifest = []
        
        for i, hl in enumerate(highlights):
            out_name = f"volleyball_highlight_{i+1:03d}.mp4"
            out_path = os.path.join(output_dir, out_name)
            
            # Comando FFmpeg otimizado (copy codec para velocidade, reencode se precisar de precisão)
            # Usamos reencode leve (libx264 crf 23) para garantir cortes precisos nos keyframes
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(hl.start),
                '-to', str(hl.end),
                '-i', self.video_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-loglevel', 'quiet', '-stats',
                out_path
            ]
            
            print(f"   Clip {i+1}: {hl.start:.2f}s -> {hl.end:.2f}s (Score: {hl.score:.2f})")
            subprocess.run(cmd, check=True)
            
            manifest.append({
                "file": out_name,
                "start": hl.start,
                "end": hl.end,
                "duration": hl.end - hl.start,
                "score": hl.score,
                "reason": hl.reason
            })
            
        # Salvar manifest JSON
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"✅ Clipes salvos em: {output_dir}")
        return manifest

    def cleanup(self):
        """Limpa arquivos temporários"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("🧹 Arquivos temporários limpos.")

def analyze_volleyball_highlights(video_path: str, output_dir: str = "highlights_output"):
    """Função principal de entrada"""
    detector = VolleyballRallyDetector(video_path)
    
    try:
        # 1. Processar e detectar
        highlights = detector.process_chunks()
        
        if not highlights:
            print("⚠️ Nenhum highlight detectado com os critérios atuais.")
            return []
            
        # 2. Gerar clipes
        manifest = detector.generate_clips(output_dir, highlights)
        
        return manifest
        
    finally:
        detector.cleanup()

# Exemplo de uso no Colab:
# if __name__ == "__main__":
#     # video_path = "/content/seu_video.mp4"
#     # analyze_volleyball_highlights(video_path)
#     pass
