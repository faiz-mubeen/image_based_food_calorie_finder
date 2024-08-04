from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import warnings
import requests
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

warnings.filterwarnings('ignore')

# Load the pre-trained Vision Transformer model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# API key for the nutrition information
api_key = 'your API key'


def identify_image(image_path):
    """Identify the food item in the image."""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    food_name = predicted_label.split(',')[0]
    return food_name

def get_calories(food_name):
    """Get the calorie information of the identified food item."""
    api_url = f'https://api.calorieninjas.com/v1/nutrition?query={food_name}'
    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    if response.status_code == requests.codes.ok:
        nutrition_info = response.json()
    else:
        nutrition_info = {"Error": response.status_code, "Message": response.text}
    return nutrition_info

def display_nutrition_info(nutrition_info):
    """Display the nutrition information in an organized table with beautiful graphics."""
    if "Error" in nutrition_info:
        return f"Error: {nutrition_info['Error']} - {nutrition_info['Message']}", None

    if len(nutrition_info['items']) == 0:
        return "No nutritional information found.", None

    # Extract data from the nutrition_info dictionary
    data = nutrition_info['items']
    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Generate the HTML table
    table_html = df.to_html(index=False, border=0, classes='table table-striped')

    # Create a bar plot for the nutritional values
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    # Select columns to plot (excluding name and serving_size_g)
    columns_to_plot = df.columns.drop(['name', 'serving_size_g'])
    df_plot = df.melt(id_vars=['name'], value_vars=columns_to_plot)
    
    # Plot the data
    sns.barplot(x='variable', y='value', hue='name', data=df_plot)
    
    # Customize the plot
    plt.title("Nutritional Values of Food Items")
    plt.xlabel("Nutritional Components")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.legend(title='Food Item')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{image_base64}"/>'
    
    return table_html, plot_html

def main_process(image_path):
    """Identify the food item and fetch its calorie information."""
    food_name = identify_image(image_path)
    nutrition_info = get_calories(food_name)
    formatted_nutrition_info, plot_html = display_nutrition_info(nutrition_info)
    return formatted_nutrition_info, plot_html

# Define the Gradio interface
def gradio_interface(image):
    formatted_nutrition_info, plot_html = main_process(image)
    return formatted_nutrition_info + plot_html

# Create the Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath"),
    outputs="html",
    title="Food Identification and Nutrition Info",
    description="Upload an image of food to get nutritional information.",
    allow_flagging="never"  # Disable flagging
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
