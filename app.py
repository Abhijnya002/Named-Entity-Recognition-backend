from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://ner-hs5bnquov-abhis-projects-2c75f009.vercel.app",
            "http://localhost:3000"
        ]
    }
})


MODEL_PATH = "abhij017/biobert-medical-ner"

print("Loading model from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f" Model loaded successfully on {device}")

def predict_entities_api(text):
    """Predict entities using the trained NER model"""
    
    # Tokenize the entire input text
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512,
        return_offsets_mapping=True,
        add_special_tokens=True
    ).to(device)
    
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0]
    
    # Convert predictions to labels
    pred_labels = [model.config.id2label[p.item()] for p in predictions]
    
    # Extract entities from BIO tags
    entities = []
    current_entity = None
    current_type = None
    current_start = None
    current_end = None
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    for idx, (token, label, offset) in enumerate(zip(tokens, pred_labels, offset_mapping)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]'] or offset[0] == offset[1]:
            if current_entity is not None:
                # Save the current entity
                entity_text = text[current_start:current_end].strip()
                entities.append({
                    'type': current_type,
                    'text': entity_text,
                    'start': current_start,
                    'end': current_end
                })
                current_entity = None
                current_type = None
            continue
        
        if label.startswith('B-'):
            # Save previous entity if exists
            if current_entity is not None:
                entity_text = text[current_start:current_end].strip()
                entities.append({
                    'type': current_type,
                    'text': entity_text,
                    'start': current_start,
                    'end': current_end
                })
            
            # Start new entity
            current_type = label[2:]  # Remove 'B-' prefix
            current_start = offset[0].item()
            current_end = offset[1].item()
            current_entity = token
            
        elif label.startswith('I-'):
            # Continue current entity
            if current_entity is not None and label[2:] == current_type:
                current_end = offset[1].item()
                current_entity += token
            else:
                # Mismatched I- tag, start new entity
                if current_entity is not None:
                    entity_text = text[current_start:current_end].strip()
                    entities.append({
                        'type': current_type,
                        'text': entity_text,
                        'start': current_start,
                        'end': current_end
                    })
                current_type = label[2:]
                current_start = offset[0].item()
                current_end = offset[1].item()
                current_entity = token
                
        else:  # 'O' tag
            # Save previous entity if exists
            if current_entity is not None:
                entity_text = text[current_start:current_end].strip()
                entities.append({
                    'type': current_type,
                    'text': entity_text,
                    'start': current_start,
                    'end': current_end
                })
                current_entity = None
                current_type = None
    
    # Don't forget the last entity
    if current_entity is not None:
        entity_text = text[current_start:current_end].strip()
        entities.append({
            'type': current_type,
            'text': entity_text,
            'start': current_start,
            'end': current_end
        })
    
    # **FILTER OUT INVALID ENTITIES**
    # Remove entities that:
    # 1. Are substrings of words (start/end mid-word)
    # 2. Are too short (< 3 chars)
    # 3. Don't start with a letter
    
    valid_entities = []
    for entity in entities:
        text_before = text[entity['start']-1:entity['start']] if entity['start'] > 0 else ' '
        text_after = text[entity['end']:entity['end']+1] if entity['end'] < len(text) else ' '
        
        # Check if entity starts/ends at word boundaries
        is_start_boundary = text_before.isspace() or text_before in '.,;:!?()[]'
        is_end_boundary = text_after.isspace() or text_after in '.,;:!?()[]'
        
        # Check if entity is valid
        entity_text = entity['text']
        is_valid = (
            is_start_boundary and  # Starts at word boundary
            is_end_boundary and    # Ends at word boundary
            len(entity_text) >= 3 and  # Minimum length
            entity_text[0].isalpha()   # Starts with letter
        )
        
        if is_valid:
            valid_entities.append({
                'type': entity['type'],
                'text': entity_text
            })
    
    return valid_entities
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Healthcare NER API',
        'status': 'running',
        'model': 'BioBERT v2',
        'f1_score': 0.8514
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics"""
    metrics = {
        'overall': {
            'precision': 0.8175,
            'recall': 0.8882,
            'f1_score': 0.8514
        },
        'chemical': {
            'precision': 0.8762,
            'recall': 0.9199,
            'f1_score': 0.8975,
            'support': 5379
        },
        'disease': {
            'precision': 0.7511,
            'recall': 0.8496,
            'f1_score': 0.7974,
            'support': 4423
        },
        'entity_distribution': {
            'chemical': 361,
            'disease': 252
        },
        'training_progress': [
            { 'epoch': 1, 'train_loss': 1.2098, 'val_loss': 1.4253 },
            { 'epoch': 2, 'train_loss': 0.5181, 'val_loss': 0.5524 },
            { 'epoch': 3, 'train_loss': 0.4572, 'val_loss': 0.5524 },
            { 'epoch': 4, 'train_loss': 0.4469, 'val_loss': 0.4426 },
            { 'epoch': 5, 'train_loss': 0.4476, 'val_loss': 0.4222 },
            { 'epoch': 6, 'train_loss': 0.4440, 'val_loss': 0.4121 },
            { 'epoch': 7, 'train_loss': 0.4438, 'val_loss': 0.4121 },
            { 'epoch': 8, 'train_loss': 0.4452, 'val_loss': 0.4040 },
            { 'epoch': 9, 'train_loss': 0.4471, 'val_loss': 0.3992 },
            { 'epoch': 10, 'train_loss': 0.4473, 'val_loss': 0.3992 },
            { 'epoch': 11, 'train_loss': 0.4488, 'val_loss': 0.3968 },
            { 'epoch': 12, 'train_loss': 0.4482, 'val_loss': 0.3956 },
            { 'epoch': 13, 'train_loss': 0.4485, 'val_loss': 0.3946 },
            { 'epoch': 14, 'train_loss': 0.4486, 'val_loss': 0.3946 },
            { 'epoch': 15, 'train_loss': 0.4497, 'val_loss': 0.3939 }
        ]
    }
    return jsonify(metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        entities = predict_entities_api(text)
        return jsonify({'entities': entities, 'success': True})
    
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': str(device)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)