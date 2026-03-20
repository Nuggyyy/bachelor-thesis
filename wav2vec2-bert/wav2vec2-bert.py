from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, Wav2Vec2BertProcessor

MODEL_NAME = "facebook/w2v-bert-2.0"


model = Wav2Vec2BertModel.from_pretrained(MODEL_NAME)
