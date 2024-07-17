# import boto3
import requests
import json
import os
import pytz
import numpy as np
from datetime import datetime
from datetime import timezone as tz
from pytz import timezone
from meteostat import Hourly

from langchain.agents import tool


def local_to_utc(datestring, tz_string="US/Central"):
    local_tz = pytz.timezone(tz_string)
    local_dt = datetime.strptime(datestring, "%m/%d/%Y %H:%M:%S")
    local_dt = local_tz.localize(local_dt)
    utc_dt = local_dt.astimezone(pytz.utc)

    return utc_dt


#   Expects start and end time to already be converted into UTC epoch seconds format.
def query_aws_sensor(sensor_id, start_time=None, end_time=None):
    aws_url = f"?sensorID={sensor_id}"

    if start_time is not None:
        aws_url += f"&start={start_time}"
    
    if end_time is not None:
        aws_url += f"&end={end_time}"
    
    response = requests.get(aws_url).json()

    if response.status_code != 200:
        raise Exception(f"Weather.gov API request did not succeed (status code not 200).  Status code: {response.status_code}")

    return response


def query_thingworx_sensor(sensor_id, oldest=False, start_time=None, end_time=None):
    headers = {"appkey": "", "Accept": "application/json", "Content-Type": "application/json"}
    thingworx_url = f"Things/{sensor_id}/Services/"

    if oldest:
        thingworx_url += 'oldest'
    else:
        thingworx_url += 'quer?'

        if start_time is not None:
            thingworx_url += f"startDate={start_time}"

        if end_time is not None:
            thingworx_url += f"&endDate={end_time}"

    response = requests.post(thingworx_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Weather.gov API request did not succeed (status code not 200).  Status code: {response.status_code}")

    return response.json()


#   TODO:  Make this a langchain agent tool.  This will be a big function that takes a string from the llm.
#          The llm will be instructed to convert the user's request into a json/csv-formated string (whichever it does better)
#          containing all possibly setable parameters:  sensor name, start time, end time.  Time values are expected to be in UTC epoch (seconds)
#          for the http request, so either the llm makes that conversion, or it just formats into a DD/MM/YYYY format and we pass that to a converter.
#          In any case, we need for it to be flexible so the user can specify as few details as possible.  Perhaps we always pass the current date to
#          llm's prompt for context, allowing the user to use relative language ("last week").
#          
#          "Thingworx expects start and end times to be in ISO format"
#
#          Once a parsable string is returned, it is parsed and passed to requests to hit our server.  We will first hit AWS and see if we 503, meaning the
#          sensorID is not located there.  If so, we check if it's in thingworx.  Eventually, we get some response of sensor data and return it, where it will
#          be passed to the agent for updated context.
#
#          Perhaps we can infer thingworx vs aws based on the sensorID itself:  if 'ENV' in sensorID then thingworx
#
#          Need to handle args for what features of interest the user has on the sensor data.  They may only want the date, temp, or humidity.
#          In this case, it may be useful to extract the data subset here so to pass a simpler context back to the agent.
#          Problem is that the keys will have different values in two diff databases.  E.g., 'temperature' in thingworx versus 'Temp' in aws.
@tool
def get_sensor_data(req_args: str) -> str:
    #   TODO:  Doc string might not allow variables in it, so I cannot put the current datetime here.  Might not update each call anyway.
    #          If so, need to figure out how to update prompt in runtime.
    """Queries sensor databases and returns requested sensor data. For action input, take values from user's input and put into JSON-formatted key value pairs with the following format: {{"sensorID": "value", "startTime": "value", "endTime": "value"}}\nWrap keys and values with quotation marks.  If no value for a key is detected, leave it blank. For startTime and endTime, enter using only numbers in this format:  month/day/year hour:minutes:seconds"""

    req_args = json.loads(req_args)

    sensor_id = req_args['sensorID']

    start_utc = None
    end_utc = None
    start_time = None
    end_time = None
    for k in req_args.keys():
        if k == 'startTime' and len(req_args[k]):
            start_utc = local_to_utc(req_args[k])
            start_time = str(int(start_utc.timestamp()))

        elif k == 'endTime' and len(req_args[k]) > 0:
            end_utc = local_to_utc(req_args[k])
            end_time = str(int(end_utc.timestamp()))

    if start_time is not None and end_time is None:
        raise Exception("Start Time must be accompanied by End Time.")

    #   TODO:  is an endTime required?  If none is provided, maybe could set it to utcnow or something.
    response = query_aws_sensor(sensor_id, start_time, end_time)

    results = None
    #   TODO:  will it return 503 if not a proper specified time?  Or if no data are within that time?  Does 503 CONFIRM that the sensor is not in AWS?
    # this is when sensorID is not found in AWS.  In this case, we can try thingworx.
    if response.status_code == 503:
        #   datetime .isoformat() will tack on the UTC timezone offset (e.g., 2024-03-04T05:59:59+00:00)
        #   thingworx seems to not accept the format because of the ...+00:00, so removing this portion from the string will be accepted.
        if start_utc is not None:
            start_time = start_utc.isoformat().split('+')[0]
        
        if end_utc is not None:
            end_time = end_utc.isoformat().split('+')[0]

        response = query_thingworx_sensor(sensor_id, start_time, end_time)

        if response.status_code == 200:
            #   Rows contains the data of thingworx.
            results = response['rows']

    elif response.status_code == 200:
        results = response
    #   Catchall for something wrong.  possibly throw exception so ask_assistant will respond accordingly.    
    else:
        pass

    #   TODO:  extract user-requested data from json to return only relevant subset to agent context.

    return json.dumps(results)


#   NOTE:  May not use this as historic data does not go further than a week or so.
#   NOTE:  KNEW station (New Orleans Lakefront Airport) was found to be closest to a pin dropped on MAF.
def query_weather_obs(start_time=None, end_time=None):
    headers = {"User-Agent": ""}
    weather_url = f"https://api.weather.gov/stations/KNEW/observations?start={start_time}+00:00&end={end_time}".replace("+", "%2B")

    response = requests.get(weather_url, headers=headers)

    results = []
    timestamps = []
    if response.status_code == 200:
        response = response.json()
        
        for f in response['features']:
            props = f['properties']

            if props['temperature']['value'] is None:
                continue

            timestamp = int(datetime.fromisoformat(props['timestamp']).timestamp())

            temp = ((9/5) * props['temperature']['value']) + 32
            #   separate timestamps array for convenient nearest time search
            timestamps.append(timestamp)

            #   NOTE:  TEMP IN CELCIUS.  Ensure all other quantities are using same units b/t weather.gov and sensors.
            results.append({"timestamp": timestamp, "temp": temp, "dewpoint": props['dewpoint']['value'], "pressure": props['barometricPressure']['value'], "relHum": props['relativeHumidity']['value']})

        timestamps = np.asarray(timestamps, dtype=np.float32)
    else:
        raise Exception(f"Weather.gov API request did not succeed (status code not 200).  Status code: {response.status_code}")

    return results, timestamps


#   TODO:  dts must be datetime objects, not utc epoch seconds or isoformat strings.
def query_meteostat(start_time=None, end_time=None):
    dataframe = Hourly("KNEW0", start_time.replace(tzinfo=None), end_time.replace(tzinfo=None)).fetch()
    dataframe = json.loads(dataframe.to_json(orient='index'))
    
    results = []
    timestamps = []
    for t in dataframe.keys():
        stamp = float(t[:10])
        timestamps.append(stamp)

        #   Celsius to Fahrenheit
        temp = ((9/5) * dataframe[t]['temp']) + 32
        results.append({"timestamp": stamp, "temp": temp, "dewpoint": dataframe[t]['dwpt'], "relhum": dataframe[t]['rhum'], "pressure": dataframe[t]['pres']})

    timestamps = np.asarray(timestamps, dtype=np.float32)

    return results, timestamps


#   NOTE:  On db state on 4/19/24, there were 1482 datapoints with a mean temporal delta of 7341 seconds between each (high variance).
#          That roughly amounts to 12 points per day.  So 50 points would be roughly 4 days.
#          
#          So maybe grab 50 points, then grab the next point, calculate time delta from last point, and convert seconds to hours.
#          That will be one sample (time delta hours is the 51st feature).  Then get the next point, recompute delta (larger than last),
#          and that's another sample.  Repeat until desired maximal forecast.  Then slide the 50 point window up one and repeat the process.
#          This should generate a lot of data.
#
#          Include humidity/other features from sensor?  YES.  Input and predict all quantitative values (especially temp, pressure, humidity/dew point)
#   TODO:  Do not include raw timestamp as feature, but perhaps YTD offset in seconds.  This will get rid of year info and allow for training data
#          to better carry over into future inference.  We still want time feature so that we have seasonal info.
def load_all_sensor_data():
    current_dt = datetime.now(timezone('UTC')).replace(microsecond=0)
    aws_sensors = ['12345678-1234-1234-1234-123456789012']

    aws_inputs = []
    aws_outputs = []
    forecast_delta = 50
    #   TODO:  To make more efficient, track oldest datetime for each sensor.  If one sensor's oldest is newer
    #          than a previous oldest, we don't need to query weather data again.  If the next sensor is older,
    #          we only need to query from that datetime to the next-oldest datetime and append to a single list.
    #          This carries over to the thingworx sensors also.
    for sensor in aws_sensors:
        #   Retrieve single datapoint to obtain the oldest time (time of first sensor reading).
        first_time = query_aws_sensor(sensor)[0]['oldestTime']

        all_results = query_aws_sensor(sensor, start_time=first_time, end_time=str(int(current_dt.timestamp())))

        # weather_results, timestamps = query_weather_obs(datetime.utcfromtimestamp(int(first_time)).isoformat(), current_dt.isoformat())
        weather_results, timestamps = query_meteostat(datetime.utcfromtimestamp(int(first_time)), current_dt)

        for i in range(forecast_delta, len(all_results) - forecast_delta):
            input_pts = all_results[i-forecast_delta:i]

            input_features = []
            for pt in input_pts:
                for k in pt.keys():
                    if k != "ID":
                        input_features.append(float(pt[k]))

                in_time = float(pt['Time'])
                diff = np.absolute(in_time - timestamps)

                weather_pt = weather_results[np.argmin(diff)]

                for k in weather_pt.keys():
                    input_features.append(weather_pt[k])

            #   Offset to generate multiple future predictions from same input.
            for j in range(1, forecast_delta+1):
                #   Hours from future prediction GT timestamp to most recent input timestamp.
                time_delta = (float(all_results[i+j]['Time']) - float(input_pts[-1]['Time'])) / 3600

                input_features.append(time_delta)
                aws_inputs.append(np.array(input_features))
                input_features.pop()

                #   temp, dewpoint, humidity
                outputs = []
                for k in all_results[i+j].keys():
                    if k not in ["ID", "Time"]:
                        outputs.append(float(all_results[i+j][k]))

                aws_outputs.append(outputs)

        np.save(f"/media/andrelongon/DATA/sensor_data/{sensor}/inputs.npy", np.array(aws_inputs))
        np.save(f"/media/andrelongon/DATA/sensor_data/{sensor}/outputs.npy", np.array(aws_outputs))

    thingworx_sensors = ['ENV-110B20', 'ENV-110B27', 'ENV-110B85']

    thingworx_inputs = []
    thingworx_outputs = []
    for sensor in thingworx_sensors:
        first_time = query_thingworx_sensor(sensor, oldest=True)['rows'][0]['timestamp']
        start_time = datetime.utcfromtimestamp(int(str(first_time)[:10])).isoformat().split('+')[0]

        all_results = query_thingworx_sensor(sensor, start_time=start_time, end_time=current_dt.isoformat().split('+')[0])['rows']

        weather_results, timestamps = query_meteostat(datetime.utcfromtimestamp(int(str(first_time)[:10])), current_dt)

        #   TODO:  to filter all results, store prev_timestamp and only append next result where its stamp is over a min delta
        #          and has temp and humidity keys.
        filtered_results = []
        delta_threshold = 7000
        #   init prev_timestamp
        prev_timestamp = None
        for result in all_results:
            if "temperature" in result.keys() and "humidity" in result.keys():
                prev_timestamp = float(str(result['timestamp'])[:10])
                filtered_results.append({"timestamp": float(str(result["timestamp"])[:10]), "temperature": result["temperature"], "humidity": result["humidity"]})
                break

        for result in all_results:
            if "temperature" in result.keys() and "humidity" in result.keys():
                if float(str(result['timestamp'])[:10]) - prev_timestamp > delta_threshold:
                    filtered_results.append({"timestamp": float(str(result["timestamp"])[:10]), "temperature": result["temperature"], "humidity": result["humidity"]})
                    prev_timestamp = float(str(result["timestamp"])[:10])

        all_results = filtered_results

        for i in range(forecast_delta, len(all_results) - forecast_delta):
            input_pts = all_results[i-forecast_delta:i]

            input_features = []
            for pt in input_pts:              
                for k in pt.keys():
                    input_features.append(pt[k])

                diff = np.absolute(pt['timestamp'] - timestamps)

                weather_pt = weather_results[np.argmin(diff)]

                for k in weather_pt.keys():
                    input_features.append(weather_pt[k])

            #   Offset to generate multiple future predictions from same input.
            for j in range(1, forecast_delta+1):
                #   Hours from future prediction GT timestamp to most recent input timestamp.
                time_delta = (all_results[i+j]['timestamp'] - input_pts[-1]['timestamp']) / 3600

                input_features.append(time_delta)
                thingworx_inputs.append(np.array(input_features))
                input_features.pop()

                #   temp, dewpoint, humidity
                outputs = []
                for k in all_results[i+j].keys():
                    if k not in ["timestamp"]:
                        outputs.append(float(all_results[i+j][k]))

                thingworx_outputs.append(outputs)

        np.save(f"/media/andrelongon/DATA/sensor_data/{sensor}/inputs.npy", np.array(thingworx_inputs))
        np.save(f"/media/andrelongon/DATA/sensor_data/{sensor}/outputs.npy", np.array(thingworx_outputs))

    #   TODO:  eventually need to normalize all the data


# load_all_sensor_data()