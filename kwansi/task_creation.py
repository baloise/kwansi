import dspy

def create_task(signature, module_name='ChainOfThought'):
    module = getattr(dspy, module_name)
    
    class Task(dspy.Module):
        def __init__(self):
            super().__init__()
            self.executor = module(signature)

        def forward(self, **kwargs):
            return self.executor(**kwargs)

    return Task
