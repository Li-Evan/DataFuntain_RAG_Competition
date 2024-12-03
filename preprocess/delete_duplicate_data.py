import json
from datetime import datetime

def deduplicate_json_file(input_file, output_file):
    seen_refs = set()
    unique_lines = []
    total_lines = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            data = json.loads(line.strip())
            ref = data.get('ref_string')

            if ref not in seen_refs:
                seen_refs.add(ref)
                unique_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)

    return total_lines, len(unique_lines)


if __name__ == '__main__':
    # 使用示例:
    input_file = '../dataset/CORAL/passage_corpus.json'
    output_file = '../dataset/CORAL/deduplicate_passage_corpus.json'
    log_file = "../log/deduplicate.txt"
    total, unique = deduplicate_json_file(input_file, output_file)
    print(f'处理前行数: {total}')
    print(f'处理后行数: {unique}')
    print(f'重复行数: {total - unique}')
    with open(log_file, 'w', encoding='utf-8') as out:
        out.write(f"Token统计报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("-" * 50 + "\n\n")
        out.write(f"\n处理前行数: {total}\n")
        out.write(f"处理后行数: {unique}\n")
        out.write(f"重复行数: {total - unique}\n")