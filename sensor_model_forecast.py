import torch
import numpy as np
from datetime import datetime
from pytz import timezone

from train_sensor_models import SensorNet
from sensors_utils import query_aws_sensor, query_thingworx_sensor, query_meteostat


#   Get past week's data, more than enough so that we more probably obtain enough timepoints (past_delta). Forecast 8 hours in future (forecast_delta).
def forecast(sensor_id, is_aws, past_delta=604800, forecast_delta=28800):
    model = SensorNet(is_aws).to("cpu").eval()
    model.load_state_dict(torch.load(f"/media/andrelongon/DATA/sensor_weights/{sensor_id}/model_1_100ep.pth"))

    current_dt = datetime.now(timezone('UTC')).replace(microsecond=0)
    start_time = str(int(current_dt.timestamp()) - past_delta)
    weather_results, timestamps = query_meteostat(datetime.utcfromtimestamp(int(start_time)), current_dt)

    past_timestamps = 50
    input_features = []
    if is_aws:
        results = query_aws_sensor(sensor_id, start_time=start_time, end_time=str(int(current_dt.timestamp())))

        if len(results) < past_timestamps:
            raise Exception("Forecast failed.  AWS query returned less than the required timestamps.")
        
        #   Iterate backwards starting closest to current_dt to collect the desired number of features then reverse input_features to obtain correct order.
        for i in range(past_timestamps):
            pt = results[-(i+1)]

            for k in pt.keys():
                if k != "ID":
                    input_features.append(float(pt[k]))

                in_time = float(pt['Time'])
                diff = np.absolute(in_time - timestamps)

                weather_pt = weather_results[np.argmin(diff)]

                for k in weather_pt.keys():
                    input_features.append(weather_pt[k])

        input_features.reverse()

    else:
        start_time = datetime.utcfromtimestamp(int(start_time)).isoformat().split('+')[0]
        results = query_thingworx_sensor(sensor_id, start_time=start_time, end_time=current_dt.isoformat().split('+')[0])['rows']

        filtered_results = []
        delta_threshold = 7000
        #   init prev_timestamp
        prev_timestamp = None
        for i in range(past_timestamps):
            result = results[-(i+1)]
            if "temperature" in result.keys() and "humidity" in result.keys():
                prev_timestamp = float(str(result['timestamp'])[:10])
                filtered_results.append({"timestamp": float(str(result["timestamp"])[:10]), "temperature": result["temperature"], "humidity": result["humidity"]})
                break

        for i in range(len(results)):
            result = results[-(i+1)]
            if "temperature" in result.keys() and "humidity" in result.keys():
                if prev_timestamp - float(str(result['timestamp'])[:10])  > delta_threshold:
                    filtered_results.append({"timestamp": float(str(result["timestamp"])[:10]), "temperature": result["temperature"], "humidity": result["humidity"]})
                    prev_timestamp = float(str(result["timestamp"])[:10])

                    if len(filtered_results) == past_timestamps:
                        break

        filtered_results.reverse()
        results = filtered_results

        if len(results) < past_timestamps:
            raise Exception("Forecast failed.  Thingworx query returned less than the required timestamps.")

        for result in results:
            for k in result.keys():
                input_features.append(result[k])

            diff = np.absolute(result['timestamp'] - timestamps)

            weather_pt = weather_results[np.argmin(diff)]

            for k in weather_pt.keys():
                input_features.append(weather_pt[k])

    input_features.append(float(forecast_delta))
        
    inputs = np.load(f"/media/andrelongon/DATA/sensor_data/{sensor_id}/inputs.npy")
    input_features = (input_features - np.mean(inputs, axis=0)) / np.std(input_features, axis=0)

    inputs = torch.unsqueeze(torch.Tensor(input_features), 0).to("cpu")

    forecast = model(inputs)
    forecast = forecast.cpu().detach().numpy()[0]

    print(f"Model forecast for sensor_id {sensor_id}:\nTemperature (F):  {forecast[0]}, Humidity (%):  {forecast[1]}")

forecast('ENV-110B20', False)