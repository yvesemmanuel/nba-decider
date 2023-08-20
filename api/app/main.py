import uvicorn
from mangum import Mangum
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas.base import (
   PredictInput,
   GambleInput
)
from services.gambler_sev import GambleService

app = FastAPI()
handler = Mangum(app)


@app.get('/')
def root():
   return {'NBADecider': 'With this API you can properly find valuable betting odds of NBA games.'}


@app.post('/predict')
def predict(input: PredictInput):
   """Endpoint to predict future NBA games outcomes."""

   service = GambleService()
   output_data = service.predict(input)

   return JSONResponse(output_data)

@app.post('/gamble')
def gamble(input: GambleInput):
   """Endpoint to make a gamble decision on NBA odds."""

   service = GambleService()
   output_data = service.gamble(input)

   return JSONResponse(output_data)


if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8080)
