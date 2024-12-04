import sys
sys.path.append('..')  # 添加父目录到Python路径
from tool.util import *
from datetime import datetime
def analyze_tokens(input_filename, output_filename):
    ranges = {
        "0-1000": 0,
        "1000-5000": 0,
        "5000-8000": 0,
        "8000+": 0
    }

    min_tokens = float('inf')
    max_tokens = 0
    total_count = 0

    with open(input_filename, 'r') as file:
        for line in file:
            # num+=1
            # if num>10 :
            #     break
            try:
                data = json.loads(line)
                ref_string = data.get('ref_string', '')
                print(ref_string)
                tokens = count_tokens(ref_string)
                total_count += 1

                min_tokens = min(min_tokens, tokens)
                max_tokens = max(max_tokens, tokens)

                if tokens < 1000:
                    ranges["0-1000"] += 1
                elif tokens < 5000:
                    ranges["1000-5000"] += 1
                elif tokens < 8000:
                    ranges["5000-8000"] += 1
                else:
                    ranges["8000+"] += 1

            except json.JSONDecodeError:
                print(f"跳过无效JSON行: {line.strip()}")

    with open(output_filename, 'w', encoding='utf-8') as out:
        out.write(f"Token统计报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("-" * 50 + "\n\n")
        out.write("区间统计:\n")
        for range_name, count in ranges.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            out.write(f"{range_name}: {count} ({percentage:.2f}%)\n")
        out.write(f"\n最小token数: {min_tokens}\n")
        out.write(f"最大token数: {max_tokens}\n")
        out.write(f"总样本数: {total_count}\n")
if __name__ == '__main__':
    file_path = "../dataset/CORAL/passage_corpus.json"
    log_file = "../log/data_analysis.txt"
    analyze_tokens(file_path,log_file)