import kagglehub

# Download latest version
path = kagglehub.dataset_download("chuneeb/deepfake-detection-dataset-2026")

print("Path to dataset files:", path)
# After the instalation, put the csv file into a folder 'data' in this project.