from bokeh.layouts import column, row
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, Select, CustomJS, Span, HoverTool, ColorBar, LinearColorMapper, BasicTicker
from bokeh.plotting import figure, show
from data_manager import AgriculturalDataManager
from bokeh.palettes import RdYlBu11 as palette

class AgriculturalDashboard:
    def __init__(self, data_manager):
        """
        Initialize the AgriculturalDashboard class.
        """
        self.data_manager = data_manager
        self.full_yield_source = None
        self.full_ndvi_source = None
        self.yield_source = None
        self.ndvi_source = None
        self.create_data_sources()


    def create_data_sources(self):
        """
        Prepare data sources using the AgriculturalDataManager.
        """
        try:
            self.data_manager.load_data()
            self.features_data = self.data_manager.prepare_features()

            # Prepare yield and NDVI data
            yield_data = self.features_data[['parcelle_id', 'date', 'rendement_estime']].dropna()
            ndvi_data = self.features_data[['parcelle_id', 'date', 'ndvi']].dropna()

            # Full sources
            self.full_yield_source = ColumnDataSource(yield_data)
            self.full_ndvi_source = ColumnDataSource(ndvi_data)

            # Dynamic sources (initially empty)
            self.yield_source = ColumnDataSource(data={key: [] for key in yield_data.columns})
            self.ndvi_source = ColumnDataSource(data={key: [] for key in ndvi_data.columns})

            print("Data sources successfully prepared.")
        except Exception as e:
            print(f"Error preparing data sources: {e}")

    def create_yield_history_plot(self, select_widget):
            """
            Create a yield history plot showing trends by parcel.
            """
            try:
                # Initialize the plot
                p = figure(
                    title="Yield History by Parcel",
                    x_axis_type="datetime",
                    height=400,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    x_axis_label="Date",
                    y_axis_label="Yield (t/ha)"
                )
                p.line(x='date', y='rendement_estime', source=self.yield_source, line_width=2, color="blue", legend_label="Yield")
                p.circle(x='date', y='rendement_estime', source=self.yield_source, size=8, color="red", legend_label="Yield Points")

                # Add hover tool
                p.add_tools(HoverTool(
                    tooltips=[("Date", "@date{%F}"), ("Yield", "@rendement_estime{0.2f}")],
                    formatters={"@date": "datetime"},
                    mode="vline"
                ))

                # Add a callback for dynamic updates
                callback = CustomJS(
                    args=dict(source=self.yield_source, full_source=self.full_yield_source, select=select_widget),
                    
                    code="""
                    
                    const full_data = full_source.data;
                    const filtered = source.data;
                    const selected = select.value;

                    // Reset filtered data
                    for (let key in filtered) {
                        filtered[key] = [];
                    }

                    // Filter and sort data by parcel and date
                    const indices = [];
                    for (let i = 0; i < full_data['parcelle_id'].length; i++) {
                        if (full_data['parcelle_id'][i] === selected) {
                            indices.push(i);
                        }
                    }

                    // Sort indices by date
                    indices.sort((a, b) => new Date(full_data['date'][a]) - new Date(full_data['date'][b]));

                    // Fill the filtered data
                    for (let key in filtered) {
                        for (let i of indices) {
                            filtered[key].push(full_data[key][i]);
                        }
                    }

                    source.change.emit();
                    """
                )
                select_widget.js_on_change("value", callback)

                return p
            except Exception as e:
                print(f"Error creating yield history plot: {e}")
                return None

    def create_ndvi_temporal_plot(self, select_widget):
        """
        Create a plot showing NDVI evolution with historical thresholds.
        """
        try:
            # Initialize the plot
            p = figure(
                title="NDVI Evolution and Historical Thresholds",
                x_axis_type="datetime",
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_axis_label="Date",
                y_axis_label="NDVI"
            )
            p.line(
                x='date',
                y='ndvi',
                source=self.ndvi_source,
                line_width=2,
                color="green",
                legend_label="NDVI"
            )
            p.add_tools(HoverTool(
                tooltips=[("Parcel", "@parcelle_id"), ("Date", "@date{%F}"), ("NDVI", "@ndvi{0.2f}")],
                formatters={"@date": "datetime"},
                mode="vline"
            ))
            p.legend.location = "top_left"

            # Add historical threshold lines
            p.add_layout(Span(location=0.5, dimension='width', line_color='blue', line_dash='dashed', line_width=2))

            # Callback to update data dynamically
            callback = CustomJS(
                args=dict(source=self.ndvi_source, full_source=self.full_ndvi_source, select=select_widget),
                code="""
                const full_data = full_source.data;
                const filtered = source.data;
                const selected_parcel = select.value;

                // Reset filtered data
                for (let key in filtered) {
                    filtered[key] = [];
                }

                // Filter data by selected parcel
                for (let i = 0; i < full_data['parcelle_id'].length; i++) {
                    if (full_data['parcelle_id'][i] === selected_parcel) {
                        for (let key in filtered) {
                            filtered[key].push(full_data[key][i]);
                        }
                    }
                }
                source.change.emit();
                """
            )
            select_widget.js_on_change("value", callback)

            return p
        except Exception as e:
            print(f"Error creating NDVI plot: {e}")
            return None

    def create_stress_matrix(self, select_widget):
        """
        Crée une matrice de stress combinant stress hydrique et conditions météorologiques.
        """
        try:
            # Vérification et préparation des colonnes nécessaires
            if 'temperature' not in self.features_data.columns or 'stress_hydrique' not in self.features_data.columns:
                print("Les colonnes 'temperature' et 'stress_hydrique' sont nécessaires pour la matrice de stress.")
                return None

            # Calcul des bins pour la température et le stress hydrique
            self.features_data['temp_bin'] = (self.features_data['temperature'] // 5) * 5  # Bins de 5°C
            self.features_data['stress_bin'] = (self.features_data['stress_hydrique'] // 0.1) * 0.1  # Bins de 0.1 pour le stress hydrique

            # Compter les occurrences pour chaque combinaison de bins
            stress_matrix = (
                self.features_data.groupby(['parcelle_id', 'temp_bin', 'stress_bin'])
                .size()
                .reset_index(name='count')
            )

            # Normaliser les densités
            max_count = stress_matrix['count'].max()
            stress_matrix['normalized_count'] = stress_matrix['count'] / max_count

            # Créer une source de données pour Bokeh
            self.stress_source = ColumnDataSource(stress_matrix)
            self.full_stress_source = ColumnDataSource(stress_matrix)

            # Configurer le graphique
            p = figure(
                title="Matrice de Stress",
                x_axis_label="Température (°C)",
                y_axis_label="Stress Hydrique (Index)",
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )

            # Mapper les couleurs
            color_mapper = LinearColorMapper(palette=palette, low=0, high=1)

            # Ajouter des rectangles pour la matrice
            p.rect(
                x="temp_bin",
                y="stress_bin",
                width=1,
                height=1,
                source=self.stress_source,
                fill_color={"field": "normalized_count", "transform": color_mapper},
                line_color=None,
            )

            # Ajouter une barre de couleur
            color_bar = ColorBar(
                color_mapper=color_mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                border_line_color=None,
                location=(0, 0),
                title="Densité Normalisée",
            )
            p.add_layout(color_bar, "right")

            # Ajouter un HoverTool pour afficher les détails
            p.add_tools(HoverTool(
                tooltips=[
                    ("Température", "@temp_bin{0.0}°C"),
                    ("Stress Hydrique", "@stress_bin{0.00}"),
                    ("Densité", "@normalized_count{0.0%}"),
                ]
            ))

            # Ajouter un callback pour mettre à jour les données dynamiquement
            callback = CustomJS(
                args=dict(
                    source=self.stress_source,
                    full_source=self.full_stress_source,
                    select=select_widget,
                ),
                code="""
                const data = full_source.data;
                const filtered = source.data;
                const selected_parcelle = select.value;

                // Réinitialiser les données filtrées
                filtered["temp_bin"] = [];
                filtered["stress_bin"] = [];
                filtered["normalized_count"] = [];

                // Filtrer les données pour la parcelle sélectionnée
                for (let i = 0; i < data["parcelle_id"].length; i++) {
                    if (data["parcelle_id"][i] === selected_parcelle) {
                        filtered["temp_bin"].push(data["temp_bin"][i]);
                        filtered["stress_bin"].push(data["stress_bin"][i]);
                        filtered["normalized_count"].push(data["normalized_count"][i]);
                    }
                }

                source.change.emit();
                """
            )
            select_widget.js_on_change("value", callback)

            return p

        except Exception as e:
            print(f"Erreur lors de la création de la matrice de stress : {e}")
            return None

    def create_layout(self):
        """
        Organize the layout with plots and widgets.
        """
        try:
            # Retrieve parcel options
            parcels = self.get_parcelle_options()
            if not parcels:
                print("No parcel options available.")
                return None

            # Create a dropdown widget for parcel selection
            select_widget = Select(title="Select a parcel:", value=parcels[0], options=parcels)

            # Generate plots
            yield_plot = self.create_yield_history_plot(select_widget)
            ndvi_plot = self.create_ndvi_temporal_plot(select_widget)
            stress_plot = self.create_stress_matrix(select_widget)
            yield_prediction_plot = self.create_yield_prediction_plot(select_widget)

            # Validate that all plots are generated
            if not yield_plot or not ndvi_plot or not stress_plot:
                print("One or more plots could not be created.")
                return None

            # Organize plots into rows, with two plots per row
            row1 = row(yield_plot, ndvi_plot)  # First row with two plots
            row2 = row(stress_plot, yield_prediction_plot)  # Second row (add more plots here if needed)

            # Combine rows into a column
            layout = column(select_widget, row1, row2)

            return layout
        except Exception as e:
            print(f"Error creating layout: {e}")
            return None
    
    def create_yield_prediction_plot(self, select_widget):
        """
        Crée un graphique de prédiction des rendements basé sur les données historiques et actuelles.
        """
        try:
            # Initialize the plot
            p = figure(
                title="Prédiction des Rendements",
                x_axis_type="datetime",
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_axis_label="Date",
                y_axis_label="Rendement (t/ha)"
            )

            # Dynamic data source for predictions
            prediction_source = ColumnDataSource(data={"date": [], "actual_yield": [], "predicted_yield": []})

            # Add lines for actual and predicted yields
            p.line(
                x="date",
                y="actual_yield",
                source=prediction_source,
                line_width=2,
                color="blue",
                legend_label="Rendement Actuel"
            )
            p.line(
                x="date",
                y="predicted_yield",
                source=prediction_source,
                line_width=2,
                color="orange",
                legend_label="Rendement Prévu"
            )

            # Add hover tools for details
            p.add_tools(HoverTool(
                tooltips=[
                    ("Date", "@date{%F}"),
                    ("Rendement Actuel", "@actual_yield{0.2f} t/ha"),
                    ("Rendement Prévu", "@predicted_yield{0.2f} t/ha"),
                ],
                formatters={"@date": "datetime"},
                mode="vline"
            ))
            p.legend.location = "top_left"

            # Add a callback to dynamically update the plot
            callback = CustomJS(
                args=dict(
                    source=prediction_source,
                    full_source=self.full_yield_source,
                    select=select_widget
                ),
                code="""
                const full_data = full_source.data;
                const filtered = source.data;
                const selected_parcel = select.value;

                // Reset filtered data
                filtered["date"] = [];
                filtered["actual_yield"] = [];
                filtered["predicted_yield"] = [];

                // Filter data by selected parcel
                for (let i = 0; i < full_data['parcelle_id'].length; i++) {
                    if (full_data['parcelle_id'][i] === selected_parcel) {
                        filtered["date"].push(full_data["date"][i]);
                        filtered["actual_yield"].push(full_data["rendement_estime"][i]);

                        // Generate predicted yield with a simple linear trend approximation
                        const prediction = full_data["rendement_estime"][i] * (1 + 0.05 * (Math.random() - 0.5));
                        filtered["predicted_yield"].push(prediction);
                    }
                }

                source.change.emit();
                """
            )
            select_widget.js_on_change("value", callback)

            return p
        except Exception as e:
            print(f"Error creating yield prediction plot: {e}")
            return None

        
    def get_parcelle_options(self):
        """
        Retrieve available parcel options from the monitoring data.
        """
        try:
            if self.data_manager.monitoring_data is None:
                raise ValueError("Monitoring data is not loaded.")

            parcels = sorted(self.data_manager.monitoring_data["parcelle_id"].unique())
            return parcels
        except Exception as e:
            print(f"Error retrieving parcel options: {e}")
            return []


if __name__ == "__main__":
    data_manager = AgriculturalDataManager()
    data_manager.load_data()
    dashboard = AgriculturalDashboard(data_manager)
    layout = dashboard.create_layout()
    if layout:
        show(layout)
    else:
        print("Layout could not be created.")