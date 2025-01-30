import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """Load the ONNX model."""
    return ort.InferenceSession(model_path)

def preprocess_image(image_path):
    """Preprocess the input image for model inference."""
    image = Image.open(image_path).resize((224, 224))  # Adjust size as needed
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    return image_array.transpose(2, 0, 1).reshape(1, 3, 224, 224)  # Change shape for model

def inference(model, image_array):
    """Run inference on the preprocessed image."""
    input_name = model.get_inputs()[0].name
    return model.run(None, {input_name: image_array})

def post_process(results):
    """Post-process the results from inference."""
    return results

def read_xlsx(file_path):
    """Read data from the specified Excel sheet and handle missing data."""
    # Read the specified sheet and fill NaN values with 0
    data = pd.read_excel(file_path, sheet_name='Tree cover loss').fillna(0)
    return data

def visualize_tree_cover_loss(data):
    """Visualize tree cover loss using seaborn and matplotlib."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Melt the DataFrame to get years and tree cover loss into a suitable format
    melted_data = data.melt(
        id_vars=['Latitude', 'Longitude', 'threshold', 'area_ha'], 
        value_vars=[col for col in data.columns if col.startswith('tc_loss_ha_')],
        var_name='Year',
        value_name='Tree Cover Loss (ha)'
    )

    # Extract the year from the Year column
    melted_data['Year'] = melted_data['Year'].str.extract(r'(\d+)').astype(int)  # Extract year as integer
    melted_data = melted_data.dropna()  # Remove any rows with NaN values if any remain

    # Create the bar plot
    sns.barplot(x='Year', y='Tree Cover Loss (ha)', data=melted_data)
    plt.title('Tree Cover Loss Over Years')
    plt.xlabel('Year')
    plt.ylabel('Tree Cover Loss (ha)')
    plt.show()

def main(image_path, model_path, xlsx_path):
    """Main function to execute the analysis."""
    # Load the model
    model = load_model(model_path)

    # Read data from the Excel file
    tree_cover_loss_data = read_xlsx(xlsx_path)

    print("\nTree Cover Loss Data:")
    print(tree_cover_loss_data.head())  # Display first few rows

    # Visualize tree cover loss data
    visualize_tree_cover_loss(tree_cover_loss_data)

    # Example for image processing (if applicable)
    if 'ImagePath' in tree_cover_loss_data.columns:
        image_path = tree_cover_loss_data['ImagePath'][0]  # Adjust based on actual structure
        image_array = preprocess_image(image_path)
        results = inference(model, image_array)
        processed_results = post_process(results)
        print("\nInference Results:")
        print(processed_results)

if __name__ == "__main__":
    model_path = "model.onnx"  # Update with actual model path
    xlsx_path = "data1.xlsx"  # Path to your Excel file
    results = main("", model_path, xlsx_path)  # Image path can be set inside the script.