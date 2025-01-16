import folium
from folium import plugins
from branca.colormap import LinearColormap
from data_manager import AgriculturalDataManager
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import webbrowser

class AgriculturalMap:
    def __init__(self, data_manager):
        """
        Initialise la carte avec le gestionnaire de données
        """
        self.data_manager = data_manager
        self.map = None
        self.yield_colormap = LinearColormap(
            colors=["red", "yellow", "green"],
            vmin=0,
            vmax=12  # Rendement maximum en tonnes/ha
        )
    
    def create_base_map(self):
        """
        Crée la carte de base avec les couches appropriées
        """
        try:
            self.data_manager.load_data()
            features = self.data_manager.prepare_features()
            avg_latitude = features['latitude'].mean()
            avg_longitude = features['longitude'].mean()
            self.map = folium.Map(
                location=[avg_latitude, avg_longitude],
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            print("Carte de base créée avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création de la carte de base : {e}")
    
    def add_yield_history_layer(self):
            """
            Ajoute une couche visualisant l'historique des rendements.
            """
            try:
                if self.map is None:
                    raise ValueError("Base map not initialized. Call create_base_map first.")

                features = self.data_manager.prepare_features()

                # Debugging: Print initial features and check for critical columns
                print("Features DataFrame (first rows):")
                print(features.head())
                required_columns = ['parcelle_id', 'latitude', 'longitude', 'rendement_estime', 'date']
                for col in required_columns:
                    if col not in features.columns:
                        raise KeyError(f"Missing required column: {col}")

                # Create 'annee' column if not present
                if 'annee' not in features.columns:
                    print("Creating 'annee' column from 'date'.")
                    features['annee'] = pd.to_datetime(features['date']).dt.year

                # Group by parcelle_id
                grouped = features.groupby('parcelle_id')
                

                # Debugging: Check groups
                print("Number of parcels:", len(grouped))

                for parcelle_id, group in grouped:
                    # Validate group data
                    if group.empty or 'latitude' not in group.columns or 'longitude' not in group.columns:
                        print(f"Skipping parcelle_id {parcelle_id} due to missing data.")
                        continue

                    # Calculate mean yield
                    mean_yield = group['rendement_estime'].mean()

                    # Utiliser la méthode modifiée pour obtenir la tendance à partir de l'historique
                    trend = self._calculate_yield_trend(parcelle_id)

                    # Generate popup content
                    popup_content = self._create_yield_popup(group, mean_yield, trend)

                    # Add to the map
                    lat = group['latitude'].mean()
                    lon = group['longitude'].mean()

                    # Validate coordinates
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        print(f"Skipping parcelle_id {parcelle_id} due to invalid coordinates: ({lat}, {lon})")
                        continue

                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=5,
                        color=self.yield_colormap(mean_yield),
                        fill=True,
                        fill_color=self.yield_colormap(mean_yield),
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(self.map)

                print("Yield history layer added successfully.")

            except Exception as e:
                print(f"Error adding yield history layer: {e}")
            
    def add_current_ndvi_layer(self):
        """
        Adds a layer visualizing current NDVI data.
        """
        try:
            if self.map is None:
                raise ValueError("Base map not initialized. Call create_base_map first.")

            features = self.data_manager.prepare_features()
            required_columns = ['parcelle_id', 'latitude', 'longitude', 'ndvi', 'culture']
            for col in required_columns:
                if col not in features.columns:
                    raise KeyError(f"Missing required column: {col}")

            ndvi_colormap = LinearColormap(
                colors=["red", "yellow", "green"],
                vmin=features['ndvi'].min(),
                vmax=features['ndvi'].max()
            )

            for _, row in features.iterrows():
                # Generate popup content using the new function
                popup_content = self._create_ndvi_popup(row)

                # Add a circle marker to the map
                folium.CircleMarker(
                    location=(row['latitude'], row['longitude']),
                    radius=5,
                    color=ndvi_colormap(row['ndvi']),
                    fill=True,
                    fill_color=ndvi_colormap(row['ndvi']),
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(self.map)

            print("Current NDVI layer added successfully.")

        except Exception as e:
            print(f"Error adding current NDVI layer: {e}")

    def add_risk_heatmap(self):
            """
            Ajoute une carte de chaleur des zones à risque
            """
            try:
                if self.map is None:
                    raise ValueError("La carte de base n'est pas initialisée. Appelez create_base_map d'abord.")
                features = self.data_manager.prepare_features()
                # Ensure columns are renamed if necessary
                features.rename(columns={
                    'lat': 'latitude',
                    'lon': 'longitude',
                    'yield': 'rendement_estime',
                    'year': 'annee'
                }, inplace=True, errors='ignore')
                risk_metrics = self.data_manager.calculate_risk_metrics(features)
                if risk_metrics is None or features is None:
                    raise ValueError("Les métriques de risque ou les données de caractéristiques manquent.")
                merged_data = pd.merge(risk_metrics, features[['parcelle_id', 'latitude', 'longitude']], on='parcelle_id', how='left')
                required_columns = ['latitude', 'longitude', 'avg_risk_index']
                if not all(col in merged_data.columns for col in required_columns):
                    raise KeyError(f"Colonnes manquantes : {', '.join(required_columns)}")
                heatmap_data = merged_data.dropna(subset=required_columns)
                heatmap_data['latitude'] += np.random.uniform(-0.0001, 0.0001, len(heatmap_data))
                heatmap_data['longitude'] += np.random.uniform(-0.0001, 0.0001, len(heatmap_data))
                min_risk = heatmap_data['avg_risk_index'].min()
                max_risk = heatmap_data['avg_risk_index'].max()
                heatmap_data['normalized_risk'] = 0.1 + (heatmap_data['avg_risk_index'] - min_risk) / (max_risk - min_risk) * 0.9
                heat_data = heatmap_data[['latitude', 'longitude', 'normalized_risk']].values.tolist()
                plugins.HeatMap(
                    heat_data,
                    name='Carte de chaleur des risques',
                    radius=15,
                    blur=15,
                    max_zoom=13,
                    min_opacity=0.3
                ).add_to(self.map)
                folium.LayerControl().add_to(self.map)
                print("Carte de chaleur des risques ajoutée avec succès.")
            except Exception as e:
                print(f"Erreur lors de l'ajout de la carte de chaleur des risques : {e}")

    def _calculate_yield_trend(self, parcelle_id):
            """
            Calculates the yield trend using linear regression.
            Utilise l'historique des rendements pour la parcelle donnée.
            """
            try:
                # Filtrer l'historique des rendements pour la parcelle donnée
                ph = self.data_manager.yield_history[self.data_manager.yield_history['parcelle_id'] == parcelle_id].copy()
                
                if ph.empty or ph['rendement_estime'].nunique() < 2:
                    # Pas assez de points de données pour une régression significative
                    return {'slope': 0, 'intercept': 0, 'variation_moyenne': 0}
                
                # S'assurer que la colonne 'annee' est au format numérique (année en entier)
                ph['date'] = ph['date'].dt.year  # Convertit en année numérique
                
                # Trier et retirer les doublons par année
                ph = ph.drop_duplicates(subset=['date']).sort_values(by='date')
                
                # Préparer les données pour la régression
                X = ph['date'].values.reshape(-1, 1)
                y = ph['rendement_estime'].values
                
                # Exécuter la régression linéaire
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                intercept = model.intercept_
                variation = slope / y.mean() if y.mean() != 0 else 0

                return {
                    'slope': slope,
                    'intercept': intercept,
                    'variation_moyenne': variation
                }
            except Exception as e:
                print(f"Error calculating yield trend for parcelle {parcelle_id}: {e}")
                return {'slope': 0, 'intercept': 0, 'variation_moyenne': 0}
    
    def _create_yield_popup(self, history, mean_yield, trend):
        """
        Creates HTML content for the yield history popup.
        """
        try:
            # Validate inputs
            if history.empty:
                raise ValueError("The history DataFrame is empty.")
            if trend is None:
                raise ValueError("Trend data is missing.")

            # Extract trend details
            slope = trend.get('slope', 0.0)
            intercept = trend.get('intercept', 0.0)
            variation = trend.get('variation_moyenne', 0.0)

            # Format recent crops
            recent_crops = self._format_recent_crops(history)

            # Create popup HTML content
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <h4 style="margin: 0; color: #2c3e50;">Parcelle ID: {history['parcelle_id'].iloc[0]}</h4>
                <p style="margin: 0; color: #34495e;">Moyenne Rendement: {mean_yield:.2f} t/ha</p>
                <p style="margin: 0; color: #34495e;">Tendance:</p>
                <ul style="margin: 0; padding-left: 15px;">
                    <li>Pente: {slope:.2f} t/ha/an</li>
                    <li>Intercept: {intercept:.2f}</li>
                    <li>Variation Moyenne: {variation:.2%}</li>
                </ul>
                <h5 style="margin-top: 10px; margin-bottom: 5px; color: #2c3e50;">Historique des Rendements:</h5>
                <ul style="margin: 0; padding-left: 15px;">
            """

            if recent_crops:
                popup_content += """
                <h5 style="margin-top: 10px; margin-bottom: 5px; color: #2c3e50;">Cultures Récentes:</h5>
                <ul style="margin: 0; padding-left: 15px;">
                """
            for crop in recent_crops:
                popup_content += f"<li>{crop}</li>"
            popup_content += "</ul>"

            # Remove duplicate years and sort history
            unique_history = history.drop_duplicates(subset=['date']).sort_values(by='date')

            # Add historical yield details
            for _, row in unique_history.iterrows():
                year = row['date']
                yield_value = row['rendement_estime']
                popup_content += f"<li>{year}: {yield_value:.2f} t/ha</li>"

            # Close the HTML tags
            popup_content += """
                </ul>
            </div>
            """

            return popup_content

        except Exception as e:
            print(f"Error creating yield popup: {e}")
            return "<div>Error creating popup content.</div>"

    def _format_recent_crops(self, history):
        """
        Formats the list of recent crops for the popup.
        """
        try:
            # Validate inputs
            if history.empty or 'culture' not in history.columns or 'date' not in history.columns:
                raise ValueError("The history DataFrame must include 'culture' and 'date'.")

            # Get the most recent crops, sorted by year
            history = history.drop_duplicates(subset=['culture', 'date']).sort_values(by='date', ascending=False)
            
            # Format crops with year
            recent_crops = history[['date', 'culture']].apply(
                lambda x: f"{x['date']}: {x['culture']}", axis=1
            ).tolist()

            return recent_crops
        except Exception as e:
            print(f"Error formatting recent crops: {e}")
            return []

    def _create_ndvi_popup(self, row):
        """
        Creates HTML content for the NDVI popup.
        """
        try:
            # Validate inputs
            required_columns = ['parcelle_id', 'latitude', 'longitude', 'ndvi']
            if not all(col in row.index for col in required_columns):
                raise ValueError("Row must include 'parcelle_id', 'latitude', 'longitude', and 'ndvi'.")
            
            

            # Create popup HTML content
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <h4 style="margin: 0; color: #2c3e50;">Parcelle ID: {row['parcelle_id']}</h4>
                <p style="margin: 0; color: #34495e;">Latitude: {row['latitude']:.5f}</p>
                <p style="margin: 0; color: #34495e;">Longitude: {row['longitude']:.5f}</p>
                <p style="margin: 0; color: #34495e;">NDVI: {row['ndvi']:.2f}</p>
            </div>
            """
            return popup_content
        except Exception as e:
            print(f"Error creating NDVI popup: {e}")
            return "<div>Error creating NDVI popup content.</div>"






if __name__ == "__main__":
    # Initialize AgriculturalDataManager and load data
    data_manager = AgriculturalDataManager()

    try:
        # Load data
        print("Loading data...")
        data_manager.load_data()

        # Initialize AgriculturalMap
        agri_map = AgriculturalMap(data_manager)

        # Create the base map
        print("Creating the base map...")
        agri_map.create_base_map()

        # Add yield history layer
        print("Adding the yield history layer...")
        agri_map.add_yield_history_layer()

        # Add current NDVI layer
        print("Adding the NDVI layer...")
        agri_map.add_current_ndvi_layer()

        # Add risk heatmap layer (optional)
        print("Adding the risk heatmap layer...")
        agri_map.add_risk_heatmap()

        # Save and open the map
        if agri_map.map:
            map_file = "agricultural_map_with_all_layers.html"
            agri_map.map.save(map_file)
            print(f"Map saved as '{map_file}'.")
            webbrowser.open(map_file)
    except Exception as e:
        print(f"An error occurred during testing: {e}")
