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

    conv_round = {
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "7+": 0
    }

    min_tokens = float('inf')
    max_tokens = 0
    min_round = 10000
    max_round = 0
    total_count = 0

    with open(input_filename, 'r') as file:
        data = json.load(file)
        for conv in data:
            conv_id = conv["conv_id"]
            turns = conv["turns"]

            # 计算对话长度相关信息
            text = ""
            for i in range(len(turns) - 1):
                single_turn = turns[i]
                text += single_turn["question"]
                text += single_turn["response"]
            text += turns[-1]["question"]
            tokens = count_tokens(text)
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

            # 计算对话轮数相关信息
            current_conv_round = len(turns)
            if current_conv_round <= 7:
                conv_round[f"{current_conv_round}"] += 1
            else:
                conv_round[f"7+"] += 1
            min_round = min(min_round,current_conv_round)
            max_round = max(max_round,current_conv_round)

    with open(output_filename, 'w', encoding='utf-8') as out:
        out.write(f"Token统计报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("-" * 50 + "\n\n")
        out.write("对话token统计:\n")
        for range_name, count in ranges.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            out.write(f"{range_name}: {count} ({percentage:.2f}%)\n")
        out.write(f"\n最小token数: {min_tokens}\n")
        out.write(f"最大token数: {max_tokens}\n")
        out.write(f"总样本数: {total_count}\n")
        out.write("\n对话轮次统计:\n")
        for conv_name, count in conv_round.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            out.write(f"{conv_name}: {count} ({percentage:.2f}%)\n")
        out.write(f"\n最小round数: {min_round}\n")
        out.write(f"最大round数: {max_round}\n")


if __name__ == '__main__':
    file_path = "../dataset/CORAL/test/a_test_conversation.json"
    log_file = "../log/multi_turn_conv_data_analysis.txt"
    analyze_tokens(file_path, log_file)
