# Loading Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("deepghs/nsfw_detect")
data = dataset["train"]

# X = data['image']  # Features
# y = data['label']  # Labels

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
