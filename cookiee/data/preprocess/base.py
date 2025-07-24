

class BasePreprocessor:
    def process(self, dataset, tokenizer, mm_plugin=None, dataset_args=None, chat_template=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __call__(self, *args, **kwds):
        raise NotImplementedError("Subclasses should implement this method.")


    def print_example(self, dataset, tokenizer, mm_plugin=None):
        raise NotImplementedError("Subclasses should implement this method.")