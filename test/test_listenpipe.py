# -*- coding: utf-8 -*-
import asyncio
import os
import wave
from loguru import logger

from voxpipeline.core.pipeline import AudioPipeline
from voxpipeline.core.datatypes import AudioChunk, TextChunk
from voxpipeline.nodes.process.fsmn_vad_node import FsmnVADNode
from voxpipeline.nodes.asr.paraformer_local import ParaformerASRNode


async def main():
    logger.info("🚀 准备启动 [VAD -> ASR] 纯听觉管线测试...")

    # ==========================================
    # 1. 实例化管线与节点
    # ==========================================
    pipeline = AudioPipeline(session_id="test_listen_001")

    vad_node = FsmnVADNode()
    asr_node = ParaformerASRNode()

    # 注册状态回调，让我们能在控制台看到内部发生跳变
    async def print_state(node_name: str, status: str, message: str):
        # 用颜色区分不同节点的状态
        color = "\033[94m" if "VAD" in node_name else "\033[92m"
        reset = "\033[0m"
        print(f"{color}[{node_name}] {status}: {message}{reset}")

    vad_node.set_state_callback(print_state)
    asr_node.set_state_callback(print_state)

    # 像拼乐高一样串联起来
    pipeline.add_node(vad_node).add_node(asr_node)

    # 启动管线后台任务
    await pipeline.run()

    # ==========================================
    # 2. 消费者任务：守在管线出口，接收最终的文本
    # ==========================================
    async def result_consumer():
        exit_queue = pipeline.exit_queue
        while True:
            chunk = await exit_queue.get()

            # 因为出口是 ASR 节点，所以出来的必须是 TextChunk
            if isinstance(chunk, TextChunk):
                if chunk.is_last:
                    logger.success("🏁 收到 EOF 信号，消费者退出。")
                    break
                if chunk.text:
                    print(f"\n🌟 \033[93m[管线最终输出] 结算出整句: {chunk.text}\033[0m\n")
            else:
                logger.error(f"类型错误！收到的是: {type(chunk)}")

    consumer_task = asyncio.create_task(result_consumer())

    # ==========================================
    # 3. 生产者任务：模拟麦克风，从本地读取 wav 并流式注入
    # ==========================================
    test_audio_path = "./locals/refs/lja.wav"  # 你之前代码里用过的测试音频

    if not os.path.exists(test_audio_path):
        logger.error(f"找不到测试音频 {test_audio_path}，请准备一段包含人声和静音的 16kHz WAV 文件！")
        await pipeline.stop()
        return

    logger.info(f"🎤 开始模拟流式发送音频: {test_audio_path}")

    with wave.open(test_audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()

        # 模拟每次发 0.1 秒的数据包
        chunk_size = int(sample_rate * sampwidth * channels * 0.1)

        while True:
            data = wf.readframes(chunk_size // sampwidth)
            if not data:
                break

            # 将字节流包装成 AudioChunk，塞入管线入口
            await pipeline.entry_queue.put(AudioChunk(data=data))

            # 模拟真实世界的时间流逝
            await asyncio.sleep(0.1)

    # ==========================================
    # 4. 收尾：发送结束信号并关闭
    # ==========================================
    logger.info("⏹ 音频流发送完毕，发送全局 EOF 信号...")
    # 发送一个 is_last=True 的空包，引发多米诺骨牌式的关闭
    await pipeline.entry_queue.put(AudioChunk(data=b"", is_last=True))

    # 等待消费者把最后一句话打印完
    await consumer_task

    # 优雅关闭管线
    await pipeline.stop()
    logger.success("🎉 测试完美结束！")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("手动中断")