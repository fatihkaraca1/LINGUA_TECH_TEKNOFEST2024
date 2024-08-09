import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Hugging Face modellerinin yolları
ner_model_path = "Karacafatih/ner_teknofest_24"
sentiment_model_path = "Karacafatih/sent_teknofest_2024"

# NER modelini ve tokenizer'ı yükle
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)

# Duygu analizi modelini ve tokenizer'ı yükle
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)

# NER ve duygu analizi pipeline'larını oluştur
ner_analyzer = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Veri modeli
class TextInput(BaseModel):
    content: str = Field(..., example="TurkNet ile ilgili yaşadığım bağlantı sorunları @SuperOnline'da da vardı. Ama @Vodafone daha iyi çözdü.")

@app.post("/analyze/", response_model=dict)
async def analyze_text(input_data: TextInput):
    text = input_data.content

    # Organizasyonları tespit et
    ner_results = ner_analyzer(text)

    if not ner_results:
        return {"message": "Metinde organizasyon bulunamadı."}

    # Tokenleri birleştir
    entities = []
    for result in ner_results:
        if result['word'].startswith('##') and entities:
            start = entities[-1]['start']
            end = result['end']
            full_word = text[start:end]
            entities[-1]['word'] = full_word
            entities[-1]['end'] = end
        else:
            entities.append(result)

    # @ ile başlayan kelimeleri ekle
    words = text.split()
    for index, word in enumerate(words):
        if word.startswith('@'):
            start = text.find(word)
            end = start + len(word)
            entities.append({
                'entity_group': 'ORG',
                'word': word,
                'start': start,
                'end': end
            })

    # Organizasyonları filtrele
    org_entities = [entity for entity in entities if entity['entity_group'] == 'ORG']

    if not org_entities:
        return {"message": "Metinde organizasyon bulunamadı."}

    # Organizasyonları sıralama
    org_entities = sorted(org_entities, key=lambda x: x['start'])

    # Organizasyon etrafındaki duygu analizini yapma
    context_window = 15  # Etrafındaki kelime sayısı

    entity_list = []
    sentiment_results = []

    i = 0
    while i < len(org_entities):
        entity = org_entities[i]
        start_idx = entity['start']
        end_idx = entity['end']

        while i + 1 < len(org_entities) and (org_entities[i + 1]['start'] - end_idx <= 1):
            end_idx = org_entities[i + 1]['end']
            i += 1

        context_start = max(0, start_idx - context_window)
        context_end = min(len(text), end_idx + context_window)
        context_text = text[context_start:context_end]

        sentiment_result = sentiment_analyzer(context_text)
        
        entity_list.append(entity['word'])
        sentiment_label = sentiment_result[0]['label']
        sentiment_text = {
            'Negative': 'olumsuz',
            'Neutral': 'nötr',
            'Positive': 'olumlu'
        }.get(sentiment_label, 'nötr')
        
        sentiment_results.append({
            "organization": entity['word'],
            "sentiment": sentiment_text
        })
        
        i += 1

    return {
        "organizations": entity_list,
        "analysis": sentiment_results
    }

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Metin Analiz Uygulaması</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f0f8ff; color: #333; }
            h1 { color: #2e8b57; }
            form { margin-top: 20px; }
            textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            input[type="submit"] { background-color: #2e8b57; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            input[type="submit"]:hover { background-color: #3cb371; }
            pre { background-color: #e6e6fa; padding: 10px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Metin Analiz Uygulaması</h1>
        <form id="analyzeForm">
            <label for="content">Metin:</label><br>
            <textarea id="content" name="content" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Analiz Et">
        </form>
        <h2>Sonuçlar:</h2>
        <pre id="output"></pre>

        <script>
            document.getElementById('analyzeForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const content = document.getElementById('content').value;
                const response = await fetch('/analyze/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ content })
                });
                const result = await response.json();
                document.getElementById('output').textContent = JSON.stringify(result, null, 2);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
