"""查看和分析端到端测试结果"""

import os
import numpy as np


def view_saved_outputs():
    """查看保存的测试输出"""
    output_dir = "test_outputs"

    print("=" * 70)
    print("查看测试结果")
    print("=" * 70)

    # 检查音频文件
    print("\n1. 音频文件:")
    original_path = os.path.join(output_dir, "original_audio.wav")
    reconstructed_path = os.path.join(output_dir, "reconstructed_audio.wav")
    generated_path = os.path.join(output_dir, "generated_audio.wav")

    if os.path.exists(original_path):
        size = os.path.getsize(original_path)
        print(f"  ✓ 原始音频: {original_path} ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ 原始音频未找到")

    if os.path.exists(reconstructed_path):
        size = os.path.getsize(reconstructed_path)
        print(f"  ✓ 重建音频: {reconstructed_path} ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ 重建音频未找到")

    if os.path.exists(generated_path):
        size = os.path.getsize(generated_path)
        print(f"  ✓ 生成的音频: {generated_path} ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ 生成的音频未找到")

    # 检查模型输出
    print("\n2. 模型输出:")
    npz_path = os.path.join(output_dir, "model_outputs.npz")

    if os.path.exists(npz_path):
        print(f"  ✓ 模型输出: {npz_path}")

        # 加载并显示
        data = np.load(npz_path)

        print("\n  内容:")
        print(f"    - text_logits: 形状 {data['text_logits'].shape}")
        print(f"    - local_hidden_states: 形状 {data['local_hidden_states'].shape}")
        print(f"    - input_ids: 形状 {data['input_ids'].shape}")

        print("\n  文本 Logits 统计:")
        logits = data['text_logits']
        print(f"    均值: {logits.mean():.4f}")
        print(f"    标准差: {logits.std():.4f}")
        print(f"    最小值: {logits.min():.4f}")
        print(f"    最大值: {logits.max():.4f}")

        # 计算 softmax 概率
        from scipy.special import softmax
        probs = softmax(logits[0, 0], axis=-1)
        top_k = 10

        # 获取 top k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]

        print(f"\n  前 {top_k} 预测的 Tokens:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            print(f"    {i+1}. Token {idx}: {prob:.6f}")

        print("\n  局部隐藏状态统计:")
        hidden = data['local_hidden_states']
        print(f"    均值: {hidden.mean():.4f}")
        print(f"    标准差: {hidden.std():.4f}")
        print(f"    最小值: {hidden.min():.4f}")
        print(f"    最大值: {hidden.max():.4f}")

        print("\n  输入 IDs 样本（前 3 个通道，前 10 个 tokens）:")
        input_ids = data['input_ids']
        for ch in range(min(3, input_ids.shape[1])):
            tokens = input_ids[0, ch, :10]
            print(f"    通道 {ch}: {tokens}")

    else:
        print(f"  ✗ 模型输出未找到")

    # 检查推理结果
    print("\n3. 完整推理结果:")
    inference_path = os.path.join(output_dir, "inference_results.npz")

    if os.path.exists(inference_path):
        print(f"  ✓ 推理结果: {inference_path}")

        # 加载并显示
        data = np.load(inference_path)

        print("\n  内容:")
        for key in data.files:
            print(f"    - {key}: 形状 {data[key].shape}")

        print("\n  生成的文本 tokens:")
        text_tokens = data['generated_text_tokens']
        print(f"    {text_tokens}")

        print("\n  生成的音频 tokens（前 3 个时间步，前 3 个通道）:")
        audio_tokens = data['generated_audio_tokens']
        for t in range(min(3, audio_tokens.shape[0])):
            print(f"    时间步 {t}: 通道0-2={audio_tokens[t, :3]}")

        print(f"\n  总共生成了 {len(text_tokens)} 个文本 tokens")
        print(f"  总共生成了 {audio_tokens.shape[0]} 个时间步的音频 tokens")
        print(f"  每个时间步有 {audio_tokens.shape[1]} 个音频通道")

    else:
        print(f"  ✗ 推理结果未找到")

    # 使用说明
    print("\n" + "=" * 70)
    print("如何播放音频文件:")
    print("=" * 70)
    if os.path.exists(original_path):
        print(f"  ffplay {original_path}")
    if os.path.exists(reconstructed_path):
        print(f"  ffplay {reconstructed_path}")
    if os.path.exists(generated_path):
        print(f"  ffplay {generated_path}")
    print("\n或使用任何音频播放器（VLC、audacity 等）")


if __name__ == "__main__":
    view_saved_outputs()
