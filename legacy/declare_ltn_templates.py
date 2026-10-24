import ltn
import torch

def at_least(trace, activity, n):
    """
    Checks if the activity occurs at least n times in the trace.
    """
    count = (trace == activity).sum(dim=-1)
    return (count >= n.view_as(count))

def response(trace, activation, target):
    activation_mask = (trace == activation)
    target_mask = (trace == target)
    has_A = activation_mask.any(dim=-1)
    _, seq_len = trace.shape
    seq_indices = torch.arange(seq_len, device=trace.device).unsqueeze(0)
    a_indices = torch.where(activation_mask, seq_indices, -1)
    last_A_idx = a_indices.max(dim=-1).values
    after_last_A = seq_indices > last_A_idx.unsqueeze(-1)
    has_B_after_last_A = (target_mask & after_last_A).any(dim=-1)
    satisfied = (~has_A) | has_B_after_last_A
    return satisfied

def precedence(trace, activation, target):
    activation_mask = (trace == activation)
    target_mask = (trace == target)
    has_B = target_mask.any(dim=-1)
    _, seq_len = trace.shape
    seq_indices = torch.arange(seq_len, device=trace.device).unsqueeze(0)
    b_indices = torch.where(target_mask, seq_indices, seq_len)
    first_B_idx = b_indices.min(dim=-1).values
    before_first_B = seq_indices < first_B_idx.unsqueeze(-1)
    has_A_before_first_B = (activation_mask & before_first_B).any(dim=-1)
    satisfied = (~has_B) | has_A_before_first_B
    return satisfied

def chainresponse(trace, activation, target):
    activation_mask = (trace == activation)
    target_mask = (trace == target)
    next_target = torch.zeros_like(target_mask)
    next_target[:, :-1] = target_mask[:, 1:]
    satisfied = (~activation_mask | next_target).all(dim=-1)
    return satisfied

def chainprecedence(trace, activation, target):
    activation_mask = (trace == activation)
    target_mask = (trace == target)
    previous_activation = torch.zeros_like(activation_mask)
    previous_activation[:, 1:] = activation_mask[:, :-1]
    satisfied = (~target_mask | previous_activation).all(dim=-1)
    return satisfied


trace_example = [[1, 2, 3, 1, 4, 1, 2], [1, 2, 3, 1, 4, 2, 3], [1, 2, 3, 3, 3, 3, 1]]
trace_example = torch.tensor(trace_example)

And = ltn.Connective(ltn.fuzzy_ops.AndProd())

act1 = ltn.Constant(torch.tensor([1]))
act2 = ltn.Constant(torch.tensor([2]))
atLeast = ltn.Predicate(func=at_least)
Response = ltn.Predicate(func=response)
Precedence = ltn.Predicate(func=precedence)
# Succession = And(Precedence, Response)
ChainResponse = ltn.Predicate(func=chainresponse)
ChainPrecedence = ltn.Predicate(func=chainprecedence)
# ChainSuccession = And(ChainPrecedence, ChainResponse)

trace_example = ltn.Variable("x", trace_example)

print(atLeast(trace_example, act1, ltn.Constant(torch.tensor([3]))))  # Should return True since activity 1 occurs 3 times
print(Response(trace_example, act1, act2))  # Should return True since after the last occurrence of 1, 4 occurs
print(Precedence(trace_example, act1, act2))  # Should return True since 1 occurs before 2 in the trace
print(ChainResponse(trace_example, act1, act2))  # Should return True since every occurrence of 1 is followed by 2