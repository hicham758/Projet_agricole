import numpy as np
import pandas as pd
import warnings

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

class AgriculturalDataManager:

    def __init__(self):
        self.monitoring_data = None
        self.weather_data = None
        self.soil_data = None
        self.yield_history = None
        self.scalar = StandardScaler()


    def load_data(self):
        try: 
            self.monitoring_data = pd.read_csv("data/monitoring_cultures.csv", parse_dates=["date"])
            self.weather_data = pd.read_csv("data/meteo_detaillee.csv", parse_dates=["date"])
            self.soil_data = pd.read_csv("data/sols.csv")
            self.yield_history = pd.read_csv("data/historique_rendements.csv", parse_dates=["date"])
        
        except FileNotFoundError as e:
            print(f"erreur: fichier introuvable. {e}")
        
        except Exception as e:
            print(f"error loading data {e}")

    
    def clean_data(self):
        
        self.weather_data['rayonnement_solaire'] = self.weather_data['rayonnement_solaire'].abs()
   
    
    def meteo_data_hourly_to_daily(self):
        try:
            self.weather_data['date'] = pd.to_datetime(self.weather_data['date'], errors='coerce')
            self.weather_data = (
                self.weather_data
                .set_index('date')
                .resample('D')
                .mean()
                .reset_index()
            )

        except Exception as e:
            print(f"Error aggregating: {e}")


    def _setup_temporal_indices(self):
        try:
            self.monitoring_data.set_index('date', inplace=True)
            self.weather_data.set_index('date', inplace=True)
            self.yield_history.set_index('date', inplace=True)

        except Exception as e:
            print(f"error setting up indexex: {e}")

    
    def prepare_features(self):
        try:
            self.monitoring_data = self.monitoring_data.sort_values(by="date")
            self.weather_data = self.weather_data.sort_values(by="date")

            data = pd.merge_asof(
                self.monitoring_data,
                self.weather_data,
                on="date",
                direction='nearest'
            )
            
            data = pd.merge(data, self.soil_data, how='left', on="parcelle_id")            
            
            data = self._enrich_with_yield_history(data)
            
            data.drop(columns=['latitude_y', 'longitude_y'], errors='ignore', inplace=True)
            data.rename(columns={'latitude_x': 'latitude', 'longitude_x': 'longitude'}, inplace=True)

            data.drop(columns=['culture_y'], errors='ignore', inplace=True)
            data.rename(columns={'culture_x': 'culture'}, inplace=True)


            data.to_csv("data/features.csv", index=False)
            print(data.columns)

            return data

        except Exception as e:
            print(f"error preparing data: {e}")


    def _enrich_with_yield_history(self, data):
        try:
            
            # Merge the yield data with the main dataset
            data = pd.merge(
                data,
                self.yield_history,
                how="left",
                on=["parcelle_id", "date"]
            )
            return data
        except Exception as e:
            print(f"error enriching yield: {e}")
            return data


    def get_temporal_patterns(self, parcelle_id):
        try:
            features = pd.read_csv("data/features_merge.csv", parse_dates=["date"])
            parcelle_data = features[features["parcelle_id"] == parcelle_id]


            if "ndvi" not in parcelle_data.columns:
                raise KeyError("NDVI column not found in the data.")

            if parcelle_data.empty:
                raise ValueError(f"No data found for parcelle_id: {parcelle_id}")

            parcelle_data.sort_values(by="date", inplace=True)
            parcelle_data.set_index("date", inplace=True)

            ndvi_series = parcelle_data["ndvi"].dropna()
            if len(ndvi_series) < 12:
                raise ValueError("Not enough data points for seasonal decomposition.")

            decomposition = seasonal_decompose(ndvi_series, model="additive", period=12)

            date_ordinal = parcelle_data.index.map(datetime.toordinal).values.reshape(-1, 1)
            ndvi_values = ndvi_series.values.reshape(-1, 1)

            trend_model = LinearRegression().fit(date_ordinal, ndvi_values)
            trend_slope = trend_model.coef_[0][0]
            trend_intercept = trend_model.intercept_[0]

            trend = {
                "pente": trend_slope,
                "intercept": trend_intercept,
                "variation_moyenne": trend_slope / ndvi_series.mean() if ndvi_series.mean() != 0 else 0
            }

            history = {
                "ndvi_trend": decomposition.trend.dropna(),
                "ndvi_seasonal": decomposition.seasonal,
                "ndvi_residual": decomposition.resid.dropna(),
                "ndvi_moving_avg": ndvi_series.rolling(window=30).mean().dropna(),
                "summary_stats": {
                    "mean_ndvi": ndvi_series.mean(),
                    "std_ndvi": ndvi_series.std(),
                    "min_ndvi": ndvi_series.min(),
                    "max_ndvi": ndvi_series.max(),
                }
            }

            return history, trend

        except Exception as e:
            print(f"Error in get_temporal_patterns: {e}")
            return None, None

    
    def calculate_risk_metrics(self, data):
        try:
            required_columns = ['parcelle_id', 'culture', 'rendement_estime', 'ph', 'matiere_organique']
            for col in required_columns:
                if col not in data.columns:
                    raise KeyError(f"Required column '{col}' is missing from the data.")

            # Normalize the required columns
            normalized_data = self.scalar.fit_transform(data[['rendement_estime', 'ph', 'matiere_organique']])

            # Calculate risk index
            data['risk_index'] = (
                0.5 * normalized_data[:, 0] +  # Weight for rendement
                0.3 * normalized_data[:, 1] +  # Weight for pH
                0.2 * normalized_data[:, 2]    # Weight for organic matter
            )

            # Assign risk categories based on thresholds
            data['risk_category'] = pd.cut(
                data['risk_index'],
                bins=[-np.inf, -1, 0, 1, np.inf],
                labels=['Très Bas', 'Bas', 'Modéré', 'Élevé']
            )

            # Group by parcelle_id and culture, aggregate risk index and risk category
            grouped_data = data.groupby(['parcelle_id', 'culture']).agg(
                avg_risk_index=('risk_index', 'mean'),
                most_frequent_risk_category=('risk_category', lambda x: x.mode()[0] if not x.mode().empty else None)
            ).reset_index()


            # Save the grouped data to a CSV file
            grouped_data.to_csv("data/grouped_risk_metrics.csv", index=False)

            return grouped_data

        except Exception as e:
            print(f"Erreur lors du calcul des métriques de risque : {e}")
            return None
        

    def analyze_yield_patterns(self, parcelle_id):
        try:
            # Extract yield history for the specified parcelle
            parcelle_yield_history = self.yield_history[
                self.yield_history['parcelle_id'] == parcelle_id
            ].copy()

            if parcelle_yield_history.empty:
                raise ValueError(f"No yield data found for parcelle_id: {parcelle_id}")

            # Ensure the data is sorted by year
            parcelle_yield_history.sort_values(by='date', inplace=True)

            # Set 'annee' as the index temporarily for analysis
            yield_series = parcelle_yield_history.set_index('date')['rendement_estime']

            # Handle missing or constant data
            if yield_series.isnull().any():
                print("Filling missing values in yield series.")
                yield_series = yield_series.interpolate()

            if yield_series.nunique() == 1:  # Check if all values are the same
                print("Adding noise to constant yield series.")
                yield_series = yield_series + np.random.normal(0, 0.1, size=len(yield_series))

            # Trend analysis using linear regression
            date_ordinal = yield_series.index.map(datetime.toordinal).values.reshape(-1, 1)
            yield_values = yield_series.values.reshape(-1, 1)

            trend_model = LinearRegression().fit(date_ordinal, yield_values)
            trend_slope = trend_model.coef_[0][0]
            trend_intercept = trend_model.intercept_[0]

            # Calculate residuals
            predicted_values = trend_model.predict(date_ordinal).flatten()
            residuals = yield_series - predicted_values

            # Compile results
            results = {
                'tendance': {
                    'pente': trend_slope,
                    'intercept': trend_intercept,
                    'variation_moyenne': trend_slope / yield_series.mean() if yield_series.mean() != 0 else 0
                },
                'residus': residuals,
                'statistiques_resume': {
                    'moyenne': yield_series.mean(),
                    'ecart_type': yield_series.std(),
                    'minimum': yield_series.min(),
                    'maximum': yield_series.max()
                }
            }

            return results

        except Exception as e:
            print(f"Erreur lors de l'analyse des patterns de rendement : {e}")
            return None
    


    
if __name__ == "__main__":
    
    data_manager = AgriculturalDataManager()


    # Load data and prepare features
    data_manager.load_data()
    data_manager.clean_data()
    data_manager.meteo_data_hourly_to_daily()

    features = data_manager.prepare_features()


    # Analyze temporal patterns for a specific parcelle
    parcelle_id = "P001"
    history, trend = data_manager.get_temporal_patterns(parcelle_id)
    risk_metric = data_manager.calculate_risk_metrics(features)
    yield_analysis = data_manager.analyze_yield_patterns(parcelle_id)

    print("========= risk metrics =========")
    print(risk_metric.head())
    print()
    print("========= Ndvi =========")
    print(f"Tendance de rendement : {trend['pente']:.4f} tonnes/ha/an")
    print(f"Variation moyenne : {trend['variation_moyenne'] * 100:.4f}%")
    print()


    print("========= rendement historique =========")
    if yield_analysis:
        print(f"Tendance de rendement : {yield_analysis['tendance']['pente']:.4f} tonnes/ha/an")
        print(f"Variation moyenne : {yield_analysis['tendance']['variation_moyenne'] * 100:.2f}%")
        print()
        print(f"Résidus : {yield_analysis['residus'].head()}")