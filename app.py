from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types
import base64
from pathlib import Path
import os



app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Global list to store pre-loaded training image parts ---
TRAINING_IMAGE_PARTS = []

# --- Function to load all training images once at startup ---
def load_all_training_images():
    train_dir = "train"
    
    # Ensure the 'train' directory exists
    if not os.path.exists(train_dir):
        print(f"Error: Training image directory '{train_dir}' not found. Please create it and place your images there.")
        return

    train_image_files = sorted([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Check if there are enough training images
    if len(train_image_files) < 10:
        print(f"Warning: Found only {len(train_image_files)} training images in '{train_dir}'. Expected 10.")
        # You might want to raise an error or exit if critical for your application
        return

    # Helper function to load a single image into a genai.types.Part
    def load_image_part(file_path):
        with open(file_path, 'rb') as f:
            image_raw = f.read()
        return types.Part.from_bytes(data=image_raw, mime_type="image/jpeg")

    print(f"Loading {len(train_image_files)} training images from '{train_dir}'...")
    for f_name in train_image_files[:10]: # Load up to 10 images
        try:
            TRAINING_IMAGE_PARTS.append(load_image_part(os.path.join(train_dir, f_name)))
            print(f"Loaded: {f_name}")
        except Exception as e:
            print(f"Error loading training image {f_name}: {e}")
    print("Finished loading training images.")

# Call the image loading function when the app starts
# This block ensures it runs within Flask's application context, if needed.
with app.app_context():
    load_all_training_images()

# --- Your existing generate function, now using pre-loaded images ---
def generate_rating(test_image_bytes):
    if len(TRAINING_IMAGE_PARTS) < 10:
        return None, "Server error: Not enough training images loaded. Please check your 'train' directory."

    # Initialize client and model inside the function or outside if preferred,
    # depending on how you manage connections. For Vertex AI, it's often fine
    # to initialize once globally or per request if context-specific.
    client = genai.Client(
      vertexai=True,
      project="trycatch-project-465608",
      location="global",
    )
    model = "gemini-2.5-flash"

    # Assign pre-loaded images directly
    image1, image2, image3, image4, image5, image6, image7, image8, image9, image10 = TRAINING_IMAGE_PARTS

    test_image_part = types.Part.from_bytes(data=test_image_bytes, mime_type="image/jpeg")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Given the images below, learn the ratings and reason associated with it properly:
                Houses with mud walls, tin roofs, grass walls are poor built and should be rated between 1-3 oit of 10.
                Houses with brick walls and only painted not plastered with poor looking roof and single storied should be rated between 4-6 out of 10.
                Houses with proper modern structure and strong roofs or balconies and plastered walls or woodden structure with multi-storied should be rated between 7-9 out of 10."""),
                
                image1,
                types.Part.from_text(text="""Rating: 2/10
        Reason: The house has a simple structure made of wooden planks and metal sheets, with an old railing and a corrugated metal roof. It sits on a mud base, which can be unstable. Overall, it looks weak and poorly built."""),
                
                image2,
                types.Part.from_text(text="""Rating: 1/10
        Reason: This house has a very weak structure made of tin sheets and worn-out plastic. The roof is patched with tarpaulin and tiles, offering little protection, and the surroundings appear cluttered and unhygienic."""),
                
                image3,
                types.Part.from_text(text="""Rating: 2/10
        Reason: This house has weak mud walls that are cracking and wearing away. The roof is made of old tiles and supported by wooden sticks, making the structure look unstable and unsafe."""),
                
                image4,
                types.Part.from_text(text="""Rating: 4/10
        Reason: This house looks colourful and artistic, the structure seems small and possibly old. The design is unique, but it may lack modern features or durability."""),
                
                image5,
                types.Part.from_text(text="""Rating: 5/10
        Reason: This house has a clean and well-maintained look with solid walls, tiled roof, and a neat entrance. However, it's quite small and may lack space or modern amenities."""),
                
                image6,
                types.Part.from_text(text="""Rating: 4/10
        Reason: Although this house is built with solid materials and has a terrace, it appears basic with limited space and minimal design. The open laundry area hints at a lack of modern facilities."""),
                
                image7,
                types.Part.from_text(text="""Rating: 8/10
        Reason: This house has solid concrete structure and elegant appearance. The sloped tiled roof helps with rainwater drainage, and the wide front porch adds to its charm. With clean lines, quality materials, and good spacing, it looks both durable and comfortable for modern living."""),
                
                image8,
                types.Part.from_text(text="""Rating: 9/10
        Reason: This house has beautiful wooden structure, spacious layout, and traditional yet elegant design. The sloped tiled roof, large veranda, and open surroundings add charm and comfort, making it both strong and aesthetically pleasing."""),
                
                image9,
                types.Part.from_text(text="""Rating: 8/10
        Reason: This house has solid concrete structure, with good utilisation of space it is feasible for modern amenities. The sleek and proper design also offers a terrace."""),
                
                image10,
                types.Part.from_text(text="""Rating : 6/10
    Reason : This house has an old solid structure made with wooden pillars. The roof is made of old tiles and supported by wooden sticks It has a neat entrance with large veranda, but it may lack modern facilities.
    Now rate the test images given below as per your learning from above."""),
                test_image_part
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        safety_settings = [types.SafetySetting(
          category="HARM_CATEGORY_HATE_SPEECH",
          threshold="OFF"
        ),types.SafetySetting(
          category="HARM_CATEGORY_DANGEROUS_CONTENT",
          threshold="OFF"
        ),types.SafetySetting(
          category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
          threshold="OFF"
        ),types.SafetySetting(
          category="HARM_CATEGORY_HARASSMENT",
          threshold="OFF"
        )],
        thinking_config=types.ThinkingConfig(
          thinking_budget=-1,
        ),
    )

    try:
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model = model,
            contents = contents,
            config = generate_content_config,
            ):
            response_text += chunk.text
        return response_text, None # Return text and no error
    except Exception as e:
        print(f"Error during content generation: {e}") # Log the error
        return None, str(e) # Return no text and error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rate_image', methods=['POST'])
def rate_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        image_bytes = file.read()
        rating_result, error = generate_rating(image_bytes)
        
        if error:
            # Return a JSON error response with a 500 status code
            return jsonify({'error': f'An error occurred during rating: {error}'}), 500
        
        # Parse the output from your model
        lines = rating_result.strip().split('\n')
        rating = "N/A"
        reason = "Could not parse reason."
        
        for line in lines:
            if line.startswith("Rating:"):
                rating = line.replace("Rating:", "").strip()
            elif line.startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()

        # Provide a more specific error if parsing failed
        if rating == "N/A" and reason == "Could not parse reason.":
            return jsonify({'error': 'Failed to parse rating and reason from model output. Model output might be malformed.', 'raw_output': rating_result}), 500

        return jsonify({'rating': rating, 'reason': reason})

if __name__ == '__main__':
    # Create 'train' and 'test' directories if they don't exist
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    
    # You'll need to place your training images (e.g., 01_poor1.jpg, etc.) inside the 'train' directory
    # And your test images for initial testing (e.g., test mid.jpeg) inside the 'test' directory
    
    app.run(debug=True) # debug=True allows for automatic reloading and better error messages