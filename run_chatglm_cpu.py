from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/chatglm3-6b"  # 本地路径

# 10个测试问题
prompts = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",  # 社会常识
    "将'我很喜欢这部电影'翻译成英语、法语、日语和西班牙语。",  # 语言能力
    "讲一个关于程序员的笑话。",  # 幽默感
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",  # 绕口令
    "一个盒子里有3个红球和2个蓝球，如果随机取出2个球，求取出的2个球都是红色的概率。",  # 逻辑推理
    "解释贝叶斯定理并给出一个实际应用的例子。",  # 数学
    "分析《红楼梦》中贾宝玉这一人物形象的特点。",  # 文学
    "如果你是一艘宇宙飞船的AI助手，描述你在遇到外星文明时会如何介绍人类。",  # 创造力
    "简述量子纠缠现象及其在量子计算中的应用。",  # 物理
    "请用Python写一个简单的快速排序算法。"  # 编程
]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"  # 自动选择 float32/float16（根据模型配置）
).eval()

# 依次处理每个问题
for i, prompt in enumerate(prompts):
    print("\n" + "="*50)
    print(f"问题{i+1}: {prompt}")
    print("="*50)
    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    
    print("\n")
