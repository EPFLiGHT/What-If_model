import urllib.request
from urllib.error import HTTPError
from wwo_hist import retrieve_hist_data # !pip install --user wwo-hist
import pickle as pkl
import pandas as pd

weather_file_path = './weather_data.csv'

with open('./weather_api_keys.txt') as file:
    api_keys = file.read().splitlines()
key_index = 0

weather_attributes = [
    'maxtempC',
    'mintempC',
    'tempC',
    'FeelsLikeC',
    'HeatIndexC',
    'humidity',
    'pressure',
    'uvIndex',
    'totalSnow_cm',
    'sunHour',
    'windspeedKmph',
    'precipMM',
]

today = pd.to_datetime("today")-pd.Timedelta(days=1)
today_str = today.strftime('%d-%b-%Y').upper()
beginning = pd.to_datetime('01 jan 2020')
beginning_str = beginning.strftime('%d-%b-%Y').upper()


def download_weather_data(path=weather_file_path):
    try:
        weather_data = pd.read_csv(path, parse_dates=['date']).set_index('date')
    except:
        weather_data = pd.DataFrame(columns = ['date', 'iso_code'] + weather_attributes).set_index('date')

    return weather_data

def get_weather_country(country, city=None, data=None, download=True):
    if(data is None and city is None):
        data = download_weather_data()

    data_country = data[data.iso_code == country.alpha_3]
    data_country = data_country.drop(['iso_code'], axis='columns')

    if (data_country.dropna().size == 0):
        print('No weather entry for ' + country.name)
        last_date = beginning-pd.Timedelta(days=1)
    else:
        data_country = data_country.sort_index(ascending=True)
        last_date = data_country.dropna().index[-1]

    if((today-last_date).days > 0 and download):
        first_date_to_dl = last_date + pd.Timedelta(days=1)
        first_date_to_dl_str = first_date_to_dl.strftime('%d-%b-%Y').upper()
        new_data_country = download_weather_country(country.alpha_2, city=None, start_date=first_date_to_dl_str, end_date=today_str)

        data_country = pd.concat([data_country, new_data_country], axis=0)

    return data_country

def download_weather_country(iso2_code, city=None, start_date='01-JAN-2020', end_date=today_str):
    if (city is None):
        iso_to_capital = pkl.load(open('../countries/iso_to_capital.pkl', 'rb'))
        city = iso_to_capital[iso2_code].replace(' ', '_')

    # Get daily data
    frequency=24
    start_date = start_date
    end_date = end_date

    global key_index
    api_key = api_keys[key_index]

    location_list = [city]

    hist_weather_data = None

    if(location_list[0]):
        done = False
        while(not done):
            try:
                done = True
                hist_weather_data = retrieve_hist_data(api_key,
                                        location_list,
                                        start_date,
                                        end_date,
                                        frequency,
                                        location_label = False,
                                        export_csv = False,
                                        store_df = True)[0]
            except urllib.error.HTTPError as err:
                done = False
                print('Key not working anymore, changing...')
                key_index += 1
                api_key = api_keys[key_index]
            except:
                print("Exception")
                done = True
                

    if(hist_weather_data is None):
        hist_weather_data = pd.DataFrame(columns = ['date'] + weather_attributes).set_index('date')
    else:
        hist_weather_data = hist_weather_data.set_index('date_time')
        hist_weather_data.index = hist_weather_data.index.rename('date')
        hist_weather_data.index = pd.to_datetime(hist_weather_data.index)
        hist_weather_data = hist_weather_data[weather_attributes]
        # Remove duplicated columns (problem in API: duplicated uvIndex)
        hist_weather_data = hist_weather_data.loc[:, ~hist_weather_data.columns.duplicated()]

    return hist_weather_data

# test = pd.read_csv('../data/merged_data/model_data_owid.csv', parse_dates=['date']).set_index('date')
# test[['iso_code']+features_weather].to_csv('../data/weather/weather_data.csv')
