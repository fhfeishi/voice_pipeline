# tests/test_full_duplex.py
import asyncio
import os
import wave
from loguru import logger

from voxpipeline.core.pipeline import AudioPipeline
from voxpipeline.core.datatypes import AudioChunk
from voxpipeline.nodes.process.fsmn_vad_node import FsmnVADNode
from voxpipeline.nodes.asr.paraformer_local import ParaformerASRNode
from voxpipeline.nodes.llm.mock_node import MockLLMNode
from voxpipeline.nodes.tts.voxcpm_local import VoxCPMTTSNode


async def main():
    logger.info("========== 启动终极全双工语音管线测试 ==========")
    pipeline = AudioPipeline(session_id="duplex_001")

    # 组装完整的四大节点大脑
    vad = FsmnVADNode()
    asr = ParaformerASRNode()
    llm = MockLLMNode()
    tts = VoxCPMTTSNode(voice_target="雷军")

    # 状态回调打印
    async def print_state(node_name, status, msg):
        print(f"\033[96m[{node_name}] {status}:\033[0m {msg}")

    for node in [vad, asr, llm, tts]:
        node.set_state_callback(print_state)
        pipeline.add_node(node)

    await pipeline.run()

    # 1. 消费者：专门负责把 TTS 生成出来的声音存盘，验证连贯性
    async def save_audio_output():
        output_wav = "test_duplex_output.wav"
        frames = []
        exit_q = pipeline.exit_queue

        while True:
            chunk: AudioChunk = await exit_q.get()
            if chunk.is_last:
                break
            if chunk.data:
                frames.append(chunk.data)

        if frames:
            with wave.open(output_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)  # VoxCPM 默认 24k
                wf.writeframes(b"".join(frames))
            logger.success(f"✅ AI 回复的音频已保存至: {output_wav}")

    consumer_task = asyncio.create_task(save_audio_output())

    # 2. 模拟前端控制器：监听 VAD 状态，触发打断
    # 这是一个极其重要的设计！管线自己不知道啥时候该打断，必须由“控制层”来调度
    async def frontend_controller():
        # 这里为了演示，我们用一个死循环来监控
        while True:
            await asyncio.sleep(0.1)
            # 真实场景中，当收到 VAD 开始信号，且 TTS 正在说话时触发
            # 这里我们在测试脚本里手动硬编码一个打断点

    # 3. 模拟麦克风发送第一句话
    test_audio = "./locals/refs/lja.wav"
    logger.info("🎤 用户发送第一句话...")
    with wave.open(test_audio, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        await pipeline.entry_queue.put(AudioChunk(data=data))
        # 加上静音包，触发 VAD 断句
        await pipeline.entry_queue.put(AudioChunk(data=b"\x00" * 32000))

        # 4. 【高潮部分】让子弹飞一会儿，然后突然打断！
    logger.info("⏳ 等待 AI 开始长篇大论 3 秒钟...")
    await asyncio.sleep(3.0)

    logger.error("⚡⚡⚡ 用户突然插话！触发系统级打断！⚡⚡⚡")
    # 触发全局清洗！你会看到 LLM 和 TTS 的日志瞬间中断
    await pipeline.interrupt_and_flush()

    logger.info("🎤 用户发送第二句短指令...")
    # 发送第二段音频（为了方便，我们还是发同一个，假装是新指令）
    with wave.open(test_audio, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        await pipeline.entry_queue.put(AudioChunk(data=data))
        await pipeline.entry_queue.put(AudioChunk(data=b"\x00" * 32000))

    await asyncio.sleep(4.0)  # 等第二句话回复完

    # 5. 关闭管线
    await pipeline.entry_queue.put(AudioChunk(data=b"", is_last=True))
    await consumer_task
    await pipeline.stop()
    logger.info("========== 测试完美结束 ==========")


if __name__ == "__main__":
    asyncio.run(main())