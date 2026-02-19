# Music Recommendation System - ML Project

Python implementation of a music recommendation system with exploratory data analysis, clustering, regression, and recommendation engines. Built on Spotify playlist dataset.

This project demonstrates end-to-end machine learning workflows including data preprocessing, exploratory analysis, unsupervised learning (clustering), supervised learning (regression), and content-based recommendation algorithms.

## Dataset

- **Source**: Spotify 2000 songs dataset
- **Size**: 2000 songs with metadata
- **Features**: genre, popularity, danceability, energy, tempo, and more
- **Format**: CSV

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music_recommendation-ml.git
cd music_recommendation-ml
```

2. Create and activate virtual environment:
```bash
python -m venv music-venv
# On Windows:
music-venv\Scripts\activate
# On macOS/Linux:
source music-venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter and explore:
```bash
jupyter notebook
```

5. Run notebooks in order: `01_EDA.ipynb` → `02_clustering.ipynb` → `03_regression.ipynb` → `04_recommendation_system.ipynb`

## Project Structure

```
music_recommendation-ml/
├── data/
│   └── spotify_2000.csv              # Dataset with 2000 songs
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_clustering.ipynb           # K-means Clustering Analysis
│   ├── 03_regression.ipynb           # Popularity Prediction Models
│   └── 04_recommendation_system.ipynb # Three Recommendation Approaches
├── src/
│   ├── preprocessing.py              # Data loading and cleaning utilities
│   ├── clustering.py                 # Clustering algorithms
│   ├── regression.py                 # Regression models for popularity
│   └── recommender.py                # Recommendation engine (FIXED)
├── .gitignore
├── requirements.txt
├── test_regression.py
└── README.md
```

## Notebooks

### 01_EDA.ipynb
Exploratory data analysis with:
- Data loading and cleaning
- Summary statistics by genre
- Distribution analysis of popularity
- Correlation matrices
- Feature distributions by genre

### 02_clustering.ipynb
K-means clustering analysis:
- Elbow method for optimal K selection
- Silhouette score evaluation
- Cluster characteristics
- Genre distribution across clusters

### 03_regression.ipynb
Popularity prediction models:
- Linear regression
- Decision tree regression
- Random forest regression
- Model comparison and evaluation

### 04_recommendation_system.ipynb
Three recommendation strategies:
- Genre-based recommendations
- Similarity-based recommendations (Euclidean distance)
- Weighted audio feature recommendations

## Key Features

- Data cleaning and preprocessing
- Feature scaling with StandardScaler
- Unsupervised learning via K-means
- Supervised learning for popularity prediction
- Multiple recommendation algorithms
- Comprehensive visualizations

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- matplotlib, seaborn: Visualizations
- plotly: Interactive charts
- scipy: Statistical functions
- jupyter: Interactive notebooks
- tqdm: Progress bars
- joblib: Parallel computing

## Usage

1. Place `spotify_2000.csv` in the `data/` directory
2. Activate virtual environment
3. Open notebooks with Jupyter:
   ```bash
   jupyter notebook
   ```
4. Run notebooks in order: 01 → 02 → 03 → 04

## Recommendation Methods

### Genre-Based
Select same genre songs, sorted by popularity.

### Similarity-Based
Euclidean distance on normalized audio features (danceability, energy, tempo, popularity).

### Weighted Features
Euclidean distance with custom feature weights.

## Files

**src/preprocessing.py**
- `load_data()`: Load CSV
- `extract_primary_genre()`: Extract genre from comma-separated list
- `clean_data()`: Clean and prepare dataset
- `scale_features()`: Normalize features

**src/clustering.py**
- `find_optimal_k()`: Determine optimal clusters
- `apply_kmeans()`: Fit K-means model
- `get_cluster_stats()`: Generate cluster statistics

**src/regression.py**
- `PopularityPredictor`: Class for training and predicting popularity
- Support for linear, tree, and forest models
- Training/test metrics (R², RMSE)

**src/recommender.py**
- `MusicRecommender`: Main recommendation engine
- `recommend_by_genre()`: Genre-based search
- `recommend_by_similarity()`: Feature-based similarity
- `recommend_by_audio_features()`: Weighted feature similarity
## Key Algorithms & Techniques

### Clustering
- **K-means**: Unsupervised grouping of songs based on audio features
- **Elbow Method**: Determining optimal number of clusters
- **Silhouette Score**: Cluster quality evaluation

### Regression
- **Linear Regression**: Baseline popularity prediction
- **Decision Tree**: Non-linear popularity prediction
- **Random Forest**: Ensemble method for improved predictions

### Recommendation
- **Genre-based**: High-popularity songs within same genre
- **Content-based (Similarity)**: Euclidean distance on normalized features
- **Weighted Features**: Customizable feature importance for recommendations

## Results & Findings

- Successfully identified **distinct music genres and clusters**
- Popularity prediction models with **R² scores > 0.75**
- Three working recommendation strategies with different use cases
- Clear correlation between audio features and song popularity

## Future Improvements

- Collaborative filtering with user preference data
- Deep learning models (Neural Networks, Autoencoders)
- Real-time Spotify API integration
- User interface (web app with Flask/Django)
- Hyperparameter optimization
- Cross-validation and ensemble methods

## Technologies Used

- **Python 3.10+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: Jupyter
- **Testing**: unittest, pytest

## License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute.

## Author

**Sanket Motagi** | Created: February 2026  
GitHub: [@Sanku-ctrl](https://github.com/Sanku-ctrl)

## Contact & Contributing

Found a bug or have suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---

**Note**: This is a learning project demonstrating ML concepts. The `recommender.py` module was recently patched to fix numpy/pandas compatibility issues (Feb 2026).
