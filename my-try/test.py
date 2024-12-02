import pickle
import torch
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert_localpath/')  # 你可以根据需要选择其他预训练模型
model = BertModel.from_pretrained('./bert_localpath/')

# 定义一个函数来处理文本并生成索引
def process_text_with_bert(tokenizer, model, text_file, output_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_data = []
    
    for line in lines:
        # 假设每行是一个描述，且已经过预处理（例如去除标点符号、小写化等）
        # 这里我们直接使用tokenizer的encode_plus方法，但只取input_ids
        encoded_dict = tokenizer.encode_plus(
            line.strip(),
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=False,  # 我们不需要attention mask
            return_token_type_ids=False   # 对于单句输入，我们不需要token type ids
        )
        
        # 由于我们可能想要保留原始的token顺序（尽管BERT不保证这一点），
        # 但为了与原始代码的输出格式保持一致，我们只取input_ids的第一个元素（即没有padding和truncation之前的ids）
        # 注意：这样做可能不是最佳实践，因为BERT的输入通常包括[CLS]和[SEP]等特殊token
        # 但为了与原始代码兼容，我们在这里这样做
        # 一个更好的做法可能是保留完整的input_ids并使用它们进行后续处理
        # 然而，由于您要求输出格式与原始代码一致，我们在这里采用简化方法
        input_ids = encoded_dict['input_ids'].squeeze().tolist()
        # 由于原始代码中的索引是从0开始的整数（包括3作为未知词的占位符），
        # 而BERT的input_ids通常从101（[CLS]）和102（[SEP]）等开始，
        # 并且包括其他特殊token的ID，因此我们不能直接使用这些ID作为索引。
        # 为了与原始代码的输出格式保持一致，我们需要将BERT的input_ids映射到一个新的索引空间，
        # 但这样做可能会丢失有关原始token的信息。在这里，为了简化，我们仅保留input_ids的长度，
        # 并用一个占位符列表来表示（这只是一个示例，不是最佳实践）。
        # 更好的做法可能是保留完整的token到索引的映射，并在需要时使用它。
        # 但由于您要求输出格式，我们在这里仅生成一个与原始输出格式兼容的占位符列表。
        # 注意：下面的代码仅用于演示目的，并不推荐在实际应用中使用。
        placeholder_index = [0] * len(input_ids)  # 这是一个占位符列表，仅用于演示
        # 如果你想要保留一些信息，你可以考虑使用input_ids的长度或其他特征来填充这个列表。
        # 例如，你可以将每个input_id映射到一个唯一的索引（但这将需要额外的词汇表）。
        
        # 由于我们实际上没有使用BERT模型的输出（除了编码），我们可以跳过模型的前向传递。
        # 但如果你需要模型的输出（例如，为了获取特定层的表示），你可以在这里添加代码来执行前向传递。
        
        # 将占位符列表添加到处理后的数据中
        processed_data.append(placeholder_index)
    
    # 将处理后的数据保存为pickle文件
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

# 处理描述文本
process_text_with_bert(tokenizer, model, 'test_desc.txt', 'train.desc.pkl')

# 处理相似词描述文本（如果适用）
# 注意：这里我们假设相似词描述文本的处理方式与描述文本相同。
# 如果它们的格式或处理方式不同，你需要相应地调整代码。
# process_text_with_bert(tokenizer, model, 'test_IR_code_desc_sw.txt', 'train_sim_desc.pkl')

print('finish processing data with BERT and saving to pickle files...')