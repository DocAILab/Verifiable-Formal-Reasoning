import json
import os
from datasets import load_dataset

def clean_fol_text(text: str) -> str:
    """清洗 FOL 文本中的常见拼写错误"""
    if not text:
        return text
    text = text.replace("Studen", "Student")
    text = text.replace("bonne", "bonnie")
    text = text.replace("bonne", "bonnie")
    return text

def convert_folio_to_unified(entry):
    label_map = {"True": "A", "False": "B", "Uncertain": "C"}
    
    premises = entry["premises"]
    if isinstance(premises, str):
        premise_list = [p.strip() for p in premises.split("\n") if p.strip()]
    else:
        premise_list = premises
    
    context = "\n".join(premise_list)
    premise_fol = clean_fol_text(entry.get("premises-FOL", ""))
    conclusion_fol = clean_fol_text(entry.get("conclusion-FOL", ""))
    
    nl2fol = {}
    if isinstance(premise_fol, list) and len(premise_list) == len(premise_fol):
        for nl, fol in zip(premise_list, premise_fol):
            nl2fol[nl] = fol
    else:
        nl2fol[context] = premise_fol if isinstance(premise_fol, str) else str(premise_fol)
    
    nl2fol[entry["conclusion"]] = conclusion_fol if isinstance(conclusion_fol, str) else str(conclusion_fol)
    
    return {
        "id": str(entry.get("example_id", "")),
        "context": context,
        "question": entry.get("conclusion", ""),
        "options": ["A) True", "B) False", "C) Uncertain"],
        "answer": label_map.get(entry.get("label", ""), "C"),
        "nl2fol": nl2fol,
        "conclusion_fol": conclusion_fol if isinstance(conclusion_fol, str) else str(conclusion_fol),
        "reasoning": "",
        "canonical_proof": [],
        "canonical_proof_reference": {
            "version": "rule_checker_v1",
            "source": "folio",
            "proof_count": 0,
            "min_proof_length": 0,
            "length_excludes_goal_binding": True,
            "length_uses_final_dependency_closure": True
        }
    }

def main():
    print("正在加载 FOLIO 数据集...")
    dataset = load_dataset('yale-nlp/FOLIO')
    
    os.makedirs('Data/FOLIO', exist_ok=True)
    
    for split in ['train', 'validation']:
        output_path = f'Data/FOLIO/converted_{split}.jsonl'
        print(f"正在转换 {split} 集...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset[split]:
                unified = convert_folio_to_unified(entry)
                f.write(json.dumps(unified, ensure_ascii=False) + '\n')
        print(f"✅ 转换完成: {output_path} (共 {len(dataset[split])} 条)")
    
    print("全部转换完成！")

if __name__ == "__main__":
    main()
