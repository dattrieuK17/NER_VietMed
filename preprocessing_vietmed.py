import json
from typing import List, Set, Dict
from pathlib import Path

def read_vietmed_json(file_path: str) -> List[Dict]:
    """
    Đọc file VietMed-NER dạng JSON
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_word_vocabulary(data: List[Dict]) -> Set[str]:
    """
    Tạo vocabulary từ tất cả các từ unique trong dataset
    """
    vocab = set()
    for sample in data:
        vocab.update(sample['words'])
    return vocab

def create_tag_set(data: List[Dict]) -> Set[str]:
    """
    Tạo tập hợp các tag unique từ trường 'labels'
    """
    tags = set()
    for sample in data:
        tags.update(sample['labels'])
    return tags

def save_dataset(output_path: str, data: List[Dict]):
    """
    Lưu dataset theo format yêu cầu:
    [words_list]\tab[tags_list]
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data:
            words = sample['words']
            labels = sample['labels']
            words_str = json.dumps(words, ensure_ascii=False)
            labels_str = json.dumps(labels, ensure_ascii=False)
            f.write(f"{words_str}\t{labels_str}\n")

def convert_vietmed_json_to_bilstm_crf(input_file: str, output_dir: str):
    """
    Chuyển đổi dữ liệu VietMed-NER JSON sang format BiLSTM-CRF
    """
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Đọc dữ liệu
    data = read_vietmed_json(input_file)
    
    # Tạo và lưu vocabulary
    vocab = create_word_vocabulary(data)
    with open(f"{output_dir}/vocab.json", 'w', encoding='utf-8') as f:
        json.dump(list(vocab), f, ensure_ascii=False, indent=2)
    
    # Tạo và lưu tags
    tags = create_tag_set(data)
    with open(f"{output_dir}/tags.json", 'w', encoding='utf-8') as f:
        json.dump(list(tags), f, ensure_ascii=False, indent=2)
    
    # Lưu dataset
    save_dataset(f"{output_dir}/dataset.txt", data)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Dữ liệu mẫu
    
    # Viết mẫu ra file để test
    # with open('train.json', 'w', encoding='utf-8') as f:
    #     json.dump(sample_data, f, ensure_ascii=False)
    
    # Chuyển đổi
    convert_vietmed_json_to_bilstm_crf('train.json', 'corpus_dir')