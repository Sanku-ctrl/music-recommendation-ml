import sys
sys.path.insert(0, 'src')
from preprocessing import load_data, clean_data
from regression import PopularityPredictor

df_raw = load_data('data/spotify_2000.csv')
df = clean_data(df_raw)

feature_cols = ['danceability', 'energy', 'tempo']
X = df[feature_cols]
y = df['popularity']

print("Testing Linear Regression...")
linear = PopularityPredictor(model_type='linear')
metrics = linear.train(X, y, test_size=0.2)
print(f"Linear R2: {metrics['test_r2']:.4f}")

print("\nTesting Tree Regression...")
tree = PopularityPredictor(model_type='tree')
metrics = tree.train(X, y, test_size=0.2)
print(f"Tree R2: {metrics['test_r2']:.4f}")

print("\nTesting Random Forest (this may take ~30 seconds)...")
forest = PopularityPredictor(model_type='forest')
metrics = forest.train(X, y, test_size=0.2)
print(f"Forest R2: {metrics['test_r2']:.4f}")

print("\nAll models trained successfully!")
