import sys
import os
sys.path.append(os.path.abspath("./CFSP"))
from CFNCFSP import CFNParser


tool = CFNParser()

sentences = [
    "根据建设部的规定，凡属于国际金融组织贷款并由国际公开招标的工程...",
    "到去年底，全区各项存款余额达七十一点六三亿元，比上年同期增长百分之四十一点七八...",
    "今天是个好日子，心想的事儿都能成。"
]

sentences2 = [
    "这个表情包来自著名动画片《飞出个未来》（Futurama）。",
    "图中的角色是Fry，他表现出一种迫不及待想购买某样东西的强烈欲望。",
    "这个表情包通常用来表达对某件新产品或服务的极度兴奋和迫切想要拥有的心情，甚至不需要更多的信息来做决定。",
    "它体现了一种冲动消费的幽默，也反映了现代社会中人们对新奇事物的追求"
]

results = tool.pipeline(sentences2)

for i, result in enumerate(results):
    print(f"Sentence {i+1} Results:")
    print(result)