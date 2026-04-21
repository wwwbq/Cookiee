from helper import get_logger
from .utils import convert_images, convert_domain
from ..spec import Role, DatasetSpec

logger = get_logger("sft-convertor")

class SftConvertor:
    def convert_alpaca(self, example, dataset_spec: DatasetSpec):
        r"""
        Converts alpaca format dataset to the standard format.
        """
        
        prompt = []
        if dataset_spec.history and isinstance(example[dataset_spec.history], list):
            for old_prompt, old_response in example[dataset_spec.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if dataset_spec.prompt and example[dataset_spec.prompt]:
            query.append(example[dataset_spec.prompt])

        if dataset_spec.query and example[dataset_spec.query]:
            query.append(example[dataset_spec.query])

        assert dataset_spec.response in example

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_spec.response]}]
        system = example[dataset_spec.system] if dataset_spec.system else None

        output = {
            "_system": system,
            "_prompt": prompt,
            "_response": response,
        }

        return output


    def convert_sharegpt(self, example, dataset_spec: DatasetSpec):
        r"""
        Converts sharegpt format dataset to the standard format.
        """

        tag_mapping = {
            dataset_spec.user_tag: Role.USER.value,
            dataset_spec.assistant_tag: Role.ASSISTANT.value,
            dataset_spec.observation_tag: Role.OBSERVATION.value,
            dataset_spec.function_tag: Role.FUNCTION.value,
            dataset_spec.system_tag: Role.SYSTEM.value,
        }

        odd_tags = (dataset_spec.user_tag, dataset_spec.observation_tag)
        even_tags = (dataset_spec.assistant_tag, dataset_spec.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages = example[dataset_spec.messages]

        # system
        if (
            dataset_spec.system_tag
            and len(messages) != 0
            and messages[0][dataset_spec.role_tag] == dataset_spec.system_tag
        ):
            system = messages[0][dataset_spec.content_tag]
            messages = messages[1:]
        else:
            system = example[dataset_spec.system] if dataset_spec.system else None

        aligned_messages = []

        broken_data = False

        # 取出messages中所有对话
        for turn_idx, message in enumerate(messages):
            if message[dataset_spec.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True

            aligned_messages.append(
                {"role": tag_mapping[message[dataset_spec.role_tag]], "content": message[dataset_spec.content_tag]}
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
    

    def __call__(self, example, dataset_spec: DatasetSpec):
        r"""
        Converts sft format dataset to the standard format.
        """
        if dataset_spec.format == "alpaca":
            outputs = self.convert_alpaca(example, dataset_spec)
        elif dataset_spec.format == "sharegpt":
            outputs = self.convert_sharegpt(example, dataset_spec)
        else:
            raise ValueError("Unsupported formatting: {}".format(dataset_spec.format))
        
        images = convert_images(example, dataset_spec)
        domain = convert_domain(example, dataset_spec)

        outputs.update({"_images": images, "_domain": domain})

        return outputs