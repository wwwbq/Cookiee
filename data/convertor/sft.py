from helper import get_logger
from .utils import Role, DatasetAttr, convert_images, convert_channel

logger = get_logger("sft-convertor")

class SftConvertor:
    def convert_alpaca(self, example, dataset_attr: DatasetAttr):
        r"""
        Converts alpaca format dataset to the standard format.
        """
        
        prompt = []
        if dataset_attr.history and isinstance(example[dataset_attr.history], list):
            for old_prompt, old_response in example[dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if dataset_attr.prompt and example[dataset_attr.prompt]:
            query.append(example[dataset_attr.prompt])

        if dataset_attr.query and example[dataset_attr.query]:
            query.append(example[dataset_attr.query])

        assert dataset_attr.response in example

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
        system = example[dataset_attr.system] if dataset_attr.system else None

        output = {
            "_system": system,
            "_prompt": prompt,
            "_response": response,
        }

        return output


    def convert_sharegpt(self, example, dataset_attr: DatasetAttr):
        r"""
        Converts sharegpt format dataset to the standard format.
        """

        tag_mapping = {
            dataset_attr.user_tag: Role.USER.value,
            dataset_attr.assistant_tag: Role.ASSISTANT.value,
            dataset_attr.observation_tag: Role.OBSERVATION.value,
            dataset_attr.function_tag: Role.FUNCTION.value,
            dataset_attr.system_tag: Role.SYSTEM.value,
        }

        odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
        even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages = example[dataset_attr.messages]

        # system
        if (
            dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
        ):
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[dataset_attr.system] if dataset_attr.system else None

        aligned_messages = []

        broken_data = False

        # 取出messages中所有对话
        for turn_idx, message in enumerate(messages):
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True

            aligned_messages.append(
                {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
            )

        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            prompt, response = [], []
        
        output = {
            "_system": system,
            "_prompt": prompt,
            "_response": response,
        }

        return output
    

    def __call__(self, example, dataset_attr: DatasetAttr):
        r"""
        Converts sft format dataset to the standard format.
        """
        if dataset_attr.formatting == "alpaca":
            outputs = self.convert_alpaca(example, dataset_attr)
        elif dataset_attr.formatting == "sharegpt":
            outputs = self.convert_sharegpt(example, dataset_attr)
        else:
            raise ValueError("Unsupported formatting: {}".format(dataset_attr.formatting))
        
        images = convert_images(example, dataset_attr)
        channel = convert_channel(example, dataset_attr)

        outputs.update({"_images": images, "_channel": channel})

        return outputs