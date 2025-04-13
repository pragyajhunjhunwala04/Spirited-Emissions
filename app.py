from flask import Flask, request, jsonify, render_template
import linear_model
import network

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    try:
        # First, ensure the model is initialized
        if not network.GPU_MODEL or not network.TIME_MODEL or not network.SEN_SCORE:
            network.train_model()
        
        # Calculate GPU usage using network.py's calc_gpu function
        gpu_usage = network.calc_gpu(prompt)
        
        # Calculate time taken using network.py's calc_time function
        time_taken = network.calc_time(prompt)
        
        # Calculate emissions using linear_model.py's calculate_carbon_emissions function
        emissions = linear_model.calculate_carbon_emissions(gpu_usage, time_taken)
        
        # Calculate driving equivalent
        driving_equivalent = emissions * 4
        
        # Return results with emissions in grams (not converting to kg)
        return jsonify({
            'gpu_percentage': f"{gpu_usage:.2f}%",
            'time_taken': f"{time_taken:.2f}s",
            'emissions': f"{emissions:.3f} g",  # Keep as grams
            'emissions_description': f"This is equivalent to using approximately {driving_equivalent/33:.2f} plastic bags."
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize models during startup
    try:
        network.train_model()
        print("Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
    
    app.run(debug=True)