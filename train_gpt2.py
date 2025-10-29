from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

import tiktoken

# 采用不放回采样
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # pytorch的切片访问，越界会截断到最大边界，访问时不会发生越界错误
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1].view(B, T)) # inputs
        y = (buf[1:].view(B, T)) # target

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1)  > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50000 BPE mergs + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12    # number of layers
    n_head: int = 12     # number of heads
    n_embd: int = 768    # embedding dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 这里的3是为了让输入进过这个线性层以后直接获得Q、K、V的组合矩阵，方便计算
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head foward to be the batch
        # nh is "number of heads", hs is "head_size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs = C = 768 channel is the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (meterialize)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # F.scaled_dot_product_attention将调用flash-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),       # 输出Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),       # 位置编码Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing scheme
        # 把wte这层和lm_head这层的权重进行绑定（tie weight），指向同一个权重矩阵
        # GPT2参考transformer论文中的做法，确保输入输出词元的embedding具有相似性
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        # 遍历每个模块，对它们的weight进行初始化，apply方法由nn.Module提供
        self.apply(self._init_weights)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Crate AdamW optimizer and use the fused version if it is availabe
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # GPT2/3的文章中没显示说明，这是看代码时发现的
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            # 残差网络会被给一个缩放因子，参考GPT-2的论文
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype = torch.long, device=device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifer
        x = self.transformer.ln_f(x)

        loss = None
        if targets is not None:
            logits = self.lm_head(x) # (B, T, vocab_size)
            # logits被展平成二维(B * T, vocab_size)，表示有B*T组样本，每组样本有vocab_size个数
            # targets被展平成一维张量(B * T, )，表示有B*T个元素的标签数组，每个元素都是[0, vocab_size-1]的整数类别
            # ignore_index=-1 表示当 targets[i] == -1 时，该样本不参与 loss 计算（常用于 padding mask）
            # 先对logits按最后一维计算softmax，可以得到一个(B * T, vocab_size)的概率矩阵p[][]
            # 根据targets取出对应的log概率，如yi=targets[i]，那么就计算log(p[i][yi])
            # 计算B*T个目标元素所在类的对数概率均值
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':  dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffers

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanil
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

import time
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

ddp_rank=0
ddp_local_rank=0
ddp_world_size=0

# 多进程执行torchrun --standalone --nproc_per_node=4 train_gpt2.py

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA or DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288   # 2**19, ~.5M, in number of tokens
B = 8       # micro batch size
T = 1024    # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

# highest: float32 = float32 * float32, high: float32 = tf32 * tf32, medium: float32 = bf16 * bf16 
torch.set_float32_matmul_precision("high")

# 50304能被128整除，通过添加不存在词元的空间来确保矩阵维度
# 尽管会增加计算量和内存消耗，但是可以适应更好的CUDA内核，运行得更快
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# betas和eps可以在GPT3的文章中找到
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# GPT-3文章指出，模型权重使用0.1作为weight_decay
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, betas = (0.9, 0.95), device_type=device)

# 1. torch.compile会查看整个模型代码，并将python解释器从中移除，转换成抽象的计算图
# 2. 分离出forward和backward计算，提前构建反向传播图，并做算子级别优化（如常量折叠、子图融合）。
# 3. 代码生成与编译执行，会把优化后的计算图翻译成更底层的高性能代码
model = torch.compile(model)

# 把模型装载到DDP容器中，在反向传播后，DDP会收集所有rank的梯度，然后进行平均，并存在每个rank上
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# GPT-3 small setting
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps=50

def get_lr(it):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # 2. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    
    # 3. in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps) # decay_ratio starts at 0 and goes to 1
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0

    # 当显存容量不足，难以支持大批次训练时，如GPT的batch size设置为0.5M
    # 就可以采用串行执行grad_accum_steps个小批次，梯度累积的方式进行模拟
    # grad_accum_steps * B * T == batch size
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y= x.to(device), y.to(device)
        

        # 混合精度计算
        # 注意FP16 高精度（10位尾数）、低动态范围（5位指数），容易溢出，BF16是低精度（7位尾数），高动态范围（8位指数，与FP32一样）
        # 因此，BF16在表示范围上与FP32一致，而如果使用FP16训练，差了很多表示范围，就要进行梯度缩放，保证在表示范围内
        # 也不是所有的操作能被autocast成bf16, softmax、layernorm、log softmax等都不会。
        # 矩阵乘法对精度变化有更多鲁棒性，因此，更多矩阵乘法能够autocast成bf16.
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            
        # 需要注意，串行小批次之后，均方误差、交叉熵误差这类带均值的计算公式，存在系数丢失的问题。
        # 比如4步分为4个1步，原本系数为1/4，现在变成了1
        loss = loss / grad_accum_steps

        # 用于打印每一步的loss
        loss_accum += loss.detach()
        # import code; code.interact(local=locals())

        if ddp:
            # model.require_backward_grad_sync为True时，会使loss.backward()触发RANK之间的梯度同步
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # we clip the global norm of the gradient at 1.0
    # 如果梯度的整体范数太大，就把它按比例缩小到一个安全范围。GPT3论文把梯度的范数缩小到了1.0，防止梯度爆炸。
    # 梯度的整体范数是一个很有用的信息，如果一直攀升，出现尖峰，就不是一个好信号。
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    # 优化器中可能有多个参数组
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000   # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)