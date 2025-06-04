from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import logging
import os
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando aplicación...")
    # En Vercel, cargar el modelo de forma síncrona en el startup
    load_model()
    yield
    # Shutdown
    logger.info("Cerrando aplicación...")

# Inicializar FastAPI
app = FastAPI(
    title="ETA Prediction API",
    description="API para predicción de tiempo estimado de llegada (ETA)",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origins - en producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)

# Modelo global
model = None

# Clase para definir los datos de entrada
class ETARequest(BaseModel):
    deviceid: str = Field(..., description="ID del dispositivo")
    direction: int = Field(..., description="Dirección del viaje")
    segment: int = Field(..., description="Segmento de la ruta")
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana (0=Lunes, 6=Domingo)")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Es fin de semana (0=No, 1=Sí)")
    start_stop: int = Field(..., description="Parada de inicio")
    end_stop: int = Field(..., description="Parada de destino")
    
    class Config:
        schema_extra = {
            "example": {
                "deviceid": "KD-262",
                "direction": 1,
                "segment": 5,
                "hour": 14,
                "day_of_week": 2,
                "month": 6,
                "is_weekend": 0,
                "start_stop": 10,
                "end_stop": 25
            }
        }

# Clase para la respuesta
class ETAResponse(BaseModel):
    eta_minutes: float = Field(..., description="ETA estimado en minutos")
    confidence: Optional[float] = Field(None, description="Nivel de confianza de la predicción")
    status: str = Field(..., description="Estado de la predicción")

def load_model(model_path: str = "lgbm_model.pkl"):
    """Cargar el modelo desde archivo .pkl"""
    global model
    if model is not None:
        return True  # Ya está cargado
        
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Modelo cargado exitosamente desde {model_path}")
            return True
        else:
            logger.error(f"Archivo del modelo no encontrado: {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        return False

def ensure_model_loaded():
    """Asegurar que el modelo esté cargado antes de hacer predicciones"""
    global model
    if model is None:
        load_model()
    return model is not None

def convert_deviceid_to_numeric(deviceid: str) -> float:
    """Convertir deviceid string a numérico"""
    try:
        # Si el deviceid es como "KD-262", extraer solo el número
        if '-' in deviceid:
            numeric_part = deviceid.split('-')[-1]
            return float(numeric_part)
        else:
            # Si ya es numérico, convertir directamente
            return float(deviceid)
    except (ValueError, IndexError):
        # Si no se puede convertir, usar un valor por defecto o hash
        return float(hash(deviceid) % 10000)

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "ETA Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=ETAResponse)
async def predict_eta(request: ETARequest):
    """
    Predecir el ETA basado en los parámetros proporcionados
    """
    if not ensure_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Verifique que el archivo .pkl existe y se haya cargado correctamente."
        )
    
    try:
        # Preparar los datos para la predicción
        features = ['deviceid', 'direction', 'segment', 'hour', 'day_of_week',
                   'month', 'is_weekend', 'start_stop', 'end_stop']
        
        # Convertir deviceid a numérico
        numeric_deviceid = convert_deviceid_to_numeric(request.deviceid)
        
        # Crear DataFrame con los datos de entrada
        input_data = pd.DataFrame([{
            'deviceid': numeric_deviceid,
            'direction': request.direction,
            'segment': request.segment,
            'hour': request.hour,
            'day_of_week': request.day_of_week,
            'month': request.month,
            'is_weekend': request.is_weekend,
            'start_stop': request.start_stop,
            'end_stop': request.end_stop
        }])
        
        # Asegurar que todos los tipos de datos sean correctos
        input_data = input_data.astype({
            'deviceid': 'float64',
            'direction': 'float64',
            'segment': 'float64',
            'hour': 'int32',
            'day_of_week': 'float64',
            'month': 'float64',
            'is_weekend': 'float64',
            'start_stop': 'float64',
            'end_stop': 'float64'
        })
        
        # Realizar la predicción
        prediction = model.predict(input_data[features])
        eta_minutes = float(prediction[0])
        
        # Verificar que la predicción sea válida
        if not np.isfinite(eta_minutes):
            raise ValueError("La predicción resultó en un valor inválido (NaN o infinito)")
        
        # Calcular confianza si el modelo lo soporta
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_data[features])
                confidence = float(np.max(proba))
                # Verificar que el valor no sea NaN o infinito
                if not np.isfinite(confidence):
                    confidence = None
            except:
                confidence = None
        elif hasattr(model, 'score'):
            try:
                # Para modelos de regresión, no usar score con una sola muestra
                # ya que R^2 no está bien definido con menos de dos muestras
                confidence = None
            except:
                confidence = None
        
        logger.info(f"Predicción realizada: ETA = {eta_minutes:.2f} minutos")
        
        return ETAResponse(
            eta_minutes=round(eta_minutes, 2),
            confidence=round(confidence, 3) if confidence is not None and np.isfinite(confidence) else None,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor durante la predicción: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_eta_batch(requests: list[ETARequest]):
    """
    Realizar predicciones en lote para múltiples solicitudes
    """
    if not ensure_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible"
        )
    
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Máximo 100 predicciones por lote"
        )
    
    try:
        features = ['deviceid', 'direction', 'segment', 'hour', 'day_of_week',
                   'month', 'is_weekend', 'start_stop', 'end_stop']
        
        # Preparar datos para todas las solicitudes
        batch_data = []
        for req in requests:
            numeric_deviceid = convert_deviceid_to_numeric(req.deviceid)
            batch_data.append({
                'deviceid': numeric_deviceid,
                'direction': req.direction,
                'segment': req.segment,
                'hour': req.hour,
                'day_of_week': req.day_of_week,
                'month': req.month,
                'is_weekend': req.is_weekend,
                'start_stop': req.start_stop,
                'end_stop': req.end_stop
            })
        
        input_df = pd.DataFrame(batch_data)
        
        # Asegurar que todos los tipos de datos sean correctos
        input_df = input_df.astype({
            'deviceid': 'float64',
            'direction': 'float64',
            'segment': 'float64',
            'hour': 'int32',
            'day_of_week': 'float64',
            'month': 'float64',
            'is_weekend': 'float64',
            'start_stop': 'float64',
            'end_stop': 'float64'
        })
        
        predictions = model.predict(input_df[features])
        
        # Crear respuestas
        responses = []
        for i, pred in enumerate(predictions):
            eta_value = float(pred)
            # Verificar que la predicción sea válida
            if not np.isfinite(eta_value):
                eta_value = 0.0  # Valor por defecto para predicciones inválidas
                
            responses.append(ETAResponse(
                eta_minutes=round(eta_value, 2),
                confidence=None,
                status="success"
            ))
        
        logger.info(f"Predicciones en lote realizadas: {len(responses)} predicciones")
        return responses
        
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción por lotes: {str(e)}"
        )

@app.post("/reload-model")
async def reload_model(model_path: str = "eta_model.pkl"):
    """
    Recargar el modelo desde archivo
    """
    try:
        if load_model(model_path):
            return {"status": "success", "message": "Modelo recargado exitosamente"}
        else:
            raise HTTPException(
                status_code=400,
                detail="No se pudo recargar el modelo"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recargar modelo: {str(e)}"
        )

# Para desarrollo local
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)