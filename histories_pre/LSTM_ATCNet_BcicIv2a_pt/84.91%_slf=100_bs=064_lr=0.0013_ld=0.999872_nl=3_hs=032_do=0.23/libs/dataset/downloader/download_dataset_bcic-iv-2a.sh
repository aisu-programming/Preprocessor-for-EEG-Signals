cd datasets
wget https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
wget https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
unzip BCICIV_2a_gdf.zip -d "BCI Competition IV 2a"
unzip true_labels.zip -d "BCI Competition IV 2a"
rm BCICIV_2a_gdf.zip
rm true_labels.zip