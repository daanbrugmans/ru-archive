import re
import string
import statistics
import math

import pandas as pd
import emot
import nltk
import contractions

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from better_profanity import profanity


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")


class Featurizer:
    def __init__(self) -> None:
        self.features = [
            "f_g_small_i_count",
            "f_g_all_caps_wordcount",
            "f_g_sentence_wo_cap_at_start",
            "f_g_sentence_lower_at_start",
            "f_g_fullstop_wo_whitespace_count",
            "f_g_a_an_error_count",
            "f_g_cont_punct_count",
            "f_i_quotation_count",
            "f_i_emoticons_count",
            "f_i_happy_emoticons_count",
            "f_i_question_count",
            "f_i_exclamation_count",
            "f_o_sentiment_score_neg",
            "f_o_sentiment_score_neu",
            "f_o_sentiment_score_pos",
            "f_o_he_she_ratio",
            "f_o_swear_word_count",
            "f_d_character_count",
            "f_d_word_count",
            "f_d_sentence_count",
            "f_d_punctuation_count",
            "f_d_digit_count",
            "f_d_uppercase_count",
            "f_d_short_word_count",
            "f_d_alphabet_count",
            "f_d_contraction_count",
            "f_d_word_without_vowels_count",
            "f_d_hapax_legomenon_count",
            "f_c_mean_word_length",
            "f_c_mean_sentence_length",
            "f_c_word_length_standard_deviation",
            "f_c_sentence_length_standard_deviation",
            "f_c_mean_word_frequency",
            "f_c_lexical_diversity_coefficient",
            "f_c_syntactic_complexity_coefficient",
            "f_c_herdans_log_type_token_richness",
        ]
        
        self.pos_tags = {
            "f_p_pos_common_noun_ratio": ["NN", "NNS"],
            "f_p_pos_proper_noun_ratio": ["NNP", "NNPS"],
            "f_p_pos_base_adjective_ratio": ["JJ",],
            "f_p_pos_comparative_adjective_ratio": ["JJR",],
            "f_p_pos_superlative_adjective_ratio": ["JJS",],
            "f_p_pos_base_adverb_ratio": ["RB",],
            "f_p_pos_comparative_adverb_ratio": ["RBR",],
            "f_p_pos_superlative_adverb_ratio": ["RBS",],
            "f_p_pos_infinitive_verb_ratio": ["VB",],
            "f_p_pos_present_tense_1st_2nd_person_verb_ratio": ["VBP",],
            "f_p_pos_present_tense_3rd_person_verb_ratio": ["VBZ",],
            "f_p_pos_past_tense_verb_ratio": ["VBD",],
            "f_p_pos_present_participle_verb_ratio": ["VBG",],
            "f_p_pos_past_participle_verb_ratio": ["VBN",],
            "f_p_pos_modal_auxiliary_verb_ratio": ["MD",],
            "f_p_pos_pronoun_ratio": ["PRP", "PRP$"],
            "f_p_pos_genetive_ratio": ["POS",],
            "f_p_pos_interjection_ratio": ["UH",],
            "f_p_pos_foreign_word_ratio": ["FW",],
            "f_p_pos_numeral_ratio": ["CD",],
            "f_p_pos_parenthesis_ratio": ["(", ")"],
        }

        self.sentiment_modelname = "cardiffnlp/twitter-roberta-base-sentiment"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            self.sentiment_modelname, cache_dir="./model_cache", model_max_length=512
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.sentiment_modelname, cache_dir="./model_cache"
        )
        self.sentiment_model.save_pretrained(save_directory="./model_cache")

        profanity.load_censor_words()

    def featurize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Featurize the texts in the dataset"""
        features_dict: dict[str, list[int | float]] = {i: [] for i in list(self.features)}
        features_dict["index"] = []
        
        for key in self.pos_tags:
            features_dict[key] = []

        for document, index in tqdm(zip(dataset["text"], dataset.index), total=len(dataset)):
            sentences: list = nltk.sent_tokenize(document)
            tokens: list = nltk.word_tokenize(document)

            features_dict["index"].append(index)
            features_dict["f_g_small_i_count"].append(self.get_small_i_count(tokens))
            features_dict["f_g_all_caps_wordcount"].append(self.get_all_caps_wordcount(tokens))
            features_dict["f_g_sentence_wo_cap_at_start"].append(self.get_sentence_wo_cap_at_start(sentences))
            features_dict["f_g_sentence_lower_at_start"].append(self.get_sentence_lower_at_start(sentences))
            features_dict["f_g_fullstop_wo_whitespace_count"].append(self.get_fullstop_wo_whitespace_count(document))
            features_dict["f_g_a_an_error_count"].append(self.get_a_an_error_count(document))
            features_dict["f_g_cont_punct_count"].append(self.get_cont_punct_count(tokens))
            features_dict["f_i_quotation_count"].append(self.get_quotation_count(document))
            features_dict["f_i_emoticons_count"].append(self.get_emoticons_count(document))
            features_dict["f_i_happy_emoticons_count"].append(self.get_happy_emoticons_count(document))
            features_dict["f_i_question_count"].append(self.get_question_count(sentences))
            features_dict["f_i_exclamation_count"].append(self.get_exclamation_count(sentences))
            neg, neu, pos = self.get_sentiment_score(document)
            features_dict["f_o_sentiment_score_neg"].append(neg)
            features_dict["f_o_sentiment_score_neu"].append(neu)
            features_dict["f_o_sentiment_score_pos"].append(pos)
            features_dict["f_o_he_she_ratio"].append(self.get_he_she_ratio(tokens))
            features_dict["f_o_swear_word_count"].append(self.get_swear_word_count(document))
            features_dict["f_d_character_count"].append(self.get_character_count(document))
            features_dict["f_d_word_count"].append(self.get_word_count(tokens))
            features_dict["f_d_sentence_count"].append(self.get_sentence_count(sentences))
            features_dict["f_d_punctuation_count"].append(self.get_punctuation_count(document))
            features_dict["f_d_digit_count"].append(self.get_digit_count(document))
            features_dict["f_d_uppercase_count"].append(self.get_uppercase_count(document))
            features_dict["f_d_short_word_count"].append(self.get_short_word_count(tokens))
            features_dict["f_d_alphabet_count"].append(self.get_alphabet_count(document))
            features_dict["f_d_contraction_count"].append(self.get_contraction_count(document))
            features_dict["f_d_word_without_vowels_count"].append(self.get_word_without_vowels_count(tokens))
            features_dict["f_d_hapax_legomenon_count"].append(self.get_hapax_legomenon_count(tokens))
            features_dict["f_c_mean_word_length"].append(self.get_mean_word_length(tokens))
            features_dict["f_c_mean_sentence_length"].append(self.get_mean_sentence_length(tokens))
            features_dict["f_c_word_length_standard_deviation"].append(self.get_word_length_standard_deviation(tokens))
            features_dict["f_c_sentence_length_standard_deviation"].append(self.get_sentence_length_standard_deviation(tokens))
            features_dict["f_c_mean_word_frequency"].append(self.get_mean_word_frequency(tokens))
            features_dict["f_c_lexical_diversity_coefficient"].append(self.get_lexical_diversity_coefficient(tokens))
            features_dict["f_c_syntactic_complexity_coefficient"].append(self.get_syntactic_complexity_coefficient(tokens))
            features_dict["f_c_herdans_log_type_token_richness"].append(self.get_herdans_log_type_token_richness(tokens))
            
            for key, value in self.get_pos_tag_ratios(tokens, self.pos_tags).items():
                features_dict[key].append(value)

        features_df = pd.DataFrame.from_dict(features_dict)
        features_df = features_df.set_index("index")
        
        return features_df

    def get_small_i_count(self, tokens: list) -> int:
        """Count words that are i (instead of I)"""
        return tokens.count("i")

    def get_all_caps_wordcount(self, tokens: list) -> int:
        """Count all words that are at least two characters long and only contain [A-Z]"""
        return sum([1 for i in tokens if re.match(r"^[A-Z]{2,}", i) is not None])

    def get_sentence_wo_cap_at_start(self, sentences: list) -> int:
        """Count all senteneces that do not start with a capital letter"""
        return sum([1 for i in sentences if re.match(r"^[A-Z].*", i) is None])

    def get_sentence_lower_at_start(self, sentences: list) -> int:
        """Count sentences that start with a small letter (possibly first something non [A-z] like ")"""
        return sum([1 for i in sentences if re.match(r"^[^A-z]*[a-z].*", i) is not None])

    def get_fullstop_wo_whitespace_count(self, document: str) -> int:
        """Count how many times a sentence ends without a space after it."""
        # A bit tricky, as nltk cannot separate sentences when there is no whitespace.
        return len(re.findall(r"[^\.][a-z]\.[A-Z][^\.]", document))

    def get_a_an_error_count(self, document: str) -> int:
        """Count how many times an a/an error occurs"""
        # Skip when next words start with h, as this depends on pronounciation
        return len(re.findall(r" an [^aeiou]", document)) + len(re.findall(r" a [aeiou]", document))

    def get_cont_punct_count(self, tokens: list) -> int:
        """Count how many times conitious punctuation occurs"""
        count = 0
        current_len = 1
        for i in range(1, len(tokens)):
            token = tokens[i]
            if token not in string.punctuation:
                if current_len > 1:
                    count += 1
                    current_len = 1
            elif tokens[i - 1] in string.punctuation:
                current_len += 1
        if current_len > 1:
            count += 1
        return count

    def get_quotation_count(self, document: str) -> int:
        """Count how many quoted text there is"""
        return len(re.findall(r"(?<=[^A-z])\".*?\"(?:[^A-z]|$)", document))

    def get_emoticons_count(self, document: str) -> int:
        """Count how many emoticons and emojis the text contains"""
        emot_obj = emot.core.emot()
        emojis = emot_obj.emoji(document)
        emoticons = emot_obj.emoticons(document)
        if "d:" in emoticons["value"]:
            emoticons["value"].remove("d:")
        return len(emoticons["value"]) + len(emojis["value"])

    def get_happy_emoticons_count(self, document: str) -> int:
        """Count how many happy emoticons and emojis the text contains"""
        return len(re.findall(r":-?\)|\(:|[xX:]D", document))

    def get_question_count(self, sentences: list) -> int:
        """Count how many sentences end in a question mark"""
        return sum([1 for i in sentences if re.search(r'\?"?$', i) is not None])

    def get_exclamation_count(self, sentences: list) -> int:
        """Count how many sentences end in a exclamation mark"""
        return sum([1 for i in sentences if re.search(r'\!"?$', i) is not None])

    def get_sentiment_score(self, document: str) -> list[float]:
        """Calculate the sentiment scores of the text: negative, neutral and positive"""
        input_ids = self.sentiment_tokenizer(document, return_tensors="pt", max_length=512, truncation=True)
        output = self.sentiment_model(**input_ids)
        return softmax(output[0][0].detach().numpy())

    def get_he_she_ratio(self, tokens: list) -> float:
        """Calculate the ratio between the words 'he' and 'she'"""
        return (tokens.count("he") + tokens.count("He") + 1) / (tokens.count("she") + tokens.count("She") + 1)

    def get_swear_word_count(self, document: str) -> int:
        """Count how many swear words are used in the text"""
        censored_text = profanity.censor(document)
        return censored_text.count("****")

    def get_character_count(self, document: str) -> int:
        """Returns the count of all characters of the given parameter string.

        Args:
            - `document`: The string to compute the amount of characters of.

        Returns:
            The count of all characters in the `document` parameter.
        """

        return len(document)

    def get_word_count(self, tokens: list) -> int:
        """Returns the count of all words that occur in the input string. 
        This count is determined using NLTK's `word_tokenize` function.
        
        Args:
            - `tokens`: The string to compute the amount of words of.
            
        Returns:
            The count of all words in the `tokens` parameter.
        """
        
        return len(tokens)

    def get_sentence_count(self, sentences: list) -> int:
        """Returns the count of all sentences that occur in the input string. 
        This count is determined using NLTK's `sent_tokenize` function.
        
        Args:
            - `sentences`: The string to compute the amount of sentences of.
            
        Returns:
            The count of all sentences in the `sentences` parameter.
        """
        return len(sentences)

    def get_punctuation_count(self, document: str) -> int:
        """Returns the count of punctuation occurrences in the input string.
        
        Args:
            - `document`: The string to compute the amount of punctuation occurrences of.
            
        Returns:
            The number of times a form of punctuation occurs in the `document` parameter.    
        """
    
        punctuation_count = 0
        
        for character in document:
            if character in list(string.punctuation):
                punctuation_count += 1
        
        return punctuation_count

    def get_digit_count(self, document: str) -> int:
        """Returns the count of individual digits occurring in the input string.
        Note that this does not mean numbers, e.g., "23" will return "2", since the number 23 consists of two digits.
        
        Args:
            - `document`: The string to compute the amount of digit occurrences of.
            
        Returns:
            The number of times a digit occurs in the `document` parameter.
        """
        digit_count = 0
        
        for character in document:
            if character in list(string.digits):
                digit_count += 1
                
        return digit_count

    def get_uppercase_count(self, document: str) -> int:
        """Returns the count of uppercase characters occurring in the input string.
        Note that only uppercase characters that are part of ASCII are supported.
        
        Args:
            - `document`: The string to compute the amount of uppercase character occurrences of.
            
        Returns:
            The number of times an uppercase character occurs in the `document` parameter.
        """
        uppercase_count = 0
        
        for character in document:
            if character in list(string.ascii_uppercase):
                uppercase_count += 1
                
        return uppercase_count

    def get_short_word_count(self, tokens: list[str], short_word_max_length: int = 4) -> int:
        """Returns the count of "short" words that occur in the `tokens` parameter.
        The cutoff point for "short" words is given by the `short_word_max_length` parameter.
        
        Args:
            - `tokens`: The string to compute the amount of "short" words of.
            - `short_word_max_length`: The maximum length of what is considered to be a "short" word. This length is inclusive.
            
        Returns:
            The number of times a "short" word occurs in the `tokens` parameter.
        """     
        short_word_list = [word for word in tokens if len(word) <= short_word_max_length]
        
        return len(short_word_list)

    def get_alphabet_count(self, document: str, include_uppercase: bool = False, include_punctuation: bool = False, include_digits: bool = False) -> int:
        """Returns the length of the alphabet of the given document. A document's alphabet is defined as all unique characters that occur in that document.
        
        Args:
            - `document`: The string for which to compute an alphabet for.
            - `include_uppercase`: A boolean used to determine whether to count uppercase characters as separate from their lowercase counterparts.
            - `include_punctuation`: A boolean used to determine whether to include punctuation in the alphabet.
            - `include_digits`: A boolean used to determine whether to include digits in the alphabet.
            
        Returns:
            The length of the alphabet of the `document` variable.
        """
        if not include_uppercase:
            document = document.lower()
        
        text_char_alphabet = set(document)
        
        if not include_punctuation:
            text_char_alphabet = {char for char in text_char_alphabet if char not in list(string.punctuation)}
            
        if not include_digits:
            text_char_alphabet = {char for char in text_char_alphabet if char not in list(string.digits)}
        
        return len(text_char_alphabet)

    def get_contraction_count(self, document: list[str], include_genetive_count: bool = True) -> int:
        """Returns the count of all contractions that occur in the given string.
        
        Args:
            - `document`: The string for which to count the number of occurring contractions.
            - `include_genetive_count`: A boolean used to determine if occurrences of the genetive should count towards the number of contractions found.
            
        Returns:
            The amount of contractions that occur in the `document` variable.
        """
        contraction_count = len(contractions.preview(document, 1))
        
        if include_genetive_count:
            tokenized_text = nltk.tokenize.word_tokenize(document)
            pos_tagged_text = nltk.tag.pos_tag(tokenized_text)
            genetive_count = len([tag for _, tag in pos_tagged_text if tag == "POS"])
            
            contraction_count += genetive_count
            
        return contraction_count

    def get_word_without_vowels_count(self, tokens: list[str], include_y_as_vowel: bool = False) -> int:
        """Returns the count of words in the input string that do not contain vowels.
        
        Args:
            - `tokens`: The string for which to count the number of words without vowels.
            - `include_y_as_vowel`: A boolean used to determine if the character "y" should be counted as a vowel.
            
        Returns:
            The number of times a word without vowels occurs in the input string.
        """
        
        word_without_vowels_count = 0
        vowels = set("aeiou")
        
        if include_y_as_vowel:
            vowels.add("y")
        
        tokens = self._remove_punctuation(tokens)
        tokens = self._remove_digits(tokens)
        
        for word in tokens:        
            if len(vowels.intersection(word)) == 0:
                word_without_vowels_count += 1
        
        return word_without_vowels_count

    def get_hapax_legomenon_count(self, tokens: list[str]) -> int:
        """Returns the count of hapax legomenon in the input text.
        A hapax legomenon is a word that occurs only once in a corpus.
        Note that for the purposes of this function, the corpus is considered to be the input text.
        
        Args:
            - `tokens`: The string for which to count the number of hapax legomenon.
            
        Returns:
            The number of hapax legomena that were found in the input text.
        """
        
        hapax_legomenon_count = 0
        
        tokens = [word.lower() for word in tokens]
            
        for word in tokens:      
            if tokens.count(word) == 1:
                hapax_legomenon_count += 1
                
        return hapax_legomenon_count
    
    def get_mean_word_length(self, tokens: list[str]) -> int:   
        word_lengths = [len(word) for word in tokens]
    
        return round(statistics.mean(word_lengths), 3)

    def get_mean_sentence_length(self, tokens: list[str]) -> int:   
        sentence_lengths = [len(sentence) for sentence in tokens]
        
        return round(statistics.mean(sentence_lengths), 3)

    def get_word_length_standard_deviation(self, tokens: list[str]) -> int:    
        word_lengths = [len(word) for word in tokens]
        
        return round(statistics.stdev(word_lengths), 3)

    def get_sentence_length_standard_deviation(self, tokens: list[str]) -> int:    
        sentence_lengths = [len(sentence) for sentence in tokens]
        
        return round(statistics.stdev(sentence_lengths), 3)

    def get_mean_word_frequency(self, tokens: list[str]) -> int:    
        word_frequencies = {}
        
        for word in tokens:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                            
        return round(statistics.mean(word_frequencies.values()), 3)

    def get_lexical_diversity_coefficient(self, tokens: list[str]) -> int:
        """From http://repository.utm.md/handle/5014/20225"""
        total_word_count = self.get_word_count(tokens)
        unique_word_count = self.get_hapax_legomenon_count(tokens)
        
        lexical_diversity_coefficient = unique_word_count / total_word_count
        
        return round(lexical_diversity_coefficient, 3)

    def get_syntactic_complexity_coefficient(self, tokens: list[str]) -> int:
        """From http://repository.utm.md/handle/5014/20225"""
        total_word_count = self.get_word_count(tokens)
        total_sentence_count = self.get_sentence_count(tokens)
        
        syntactic_complexity_coefficient = 1 - total_sentence_count / total_word_count
        
        return round(syntactic_complexity_coefficient, 3)

    def get_herdans_log_type_token_richness(self, tokens: list[str]) -> int:
        """From https://pubs.asha.org/doi/abs/10.1044/jshr.3203.536"""
        total_word_count = self.get_word_count(tokens)
        unique_word_count = self.get_hapax_legomenon_count(tokens)
        
        herdans_log_type_token_richness = math.log(unique_word_count) / math.log(total_word_count)
        
        return round(herdans_log_type_token_richness, 3)
    
    def get_pos_tag_ratios(self, tokens: list[str], pos_tag_groups: dict[str, list[str]]) -> dict[str, int]:
        total_word_count = len(tokens)
        pos_tag_ratios = {}
        
        text_pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]
        
        for key, value in pos_tag_groups.items():
            pos_tag_ratios[key] = 0
            
            for pos_tag in value:
                pos_tag_ratios[key] += text_pos_tags.count(pos_tag)
                
            pos_tag_ratios[key] = round(pos_tag_ratios[key] / total_word_count, 3)
        
        return pos_tag_ratios

    # Helper functions
    def _remove_punctuation(self, tokens: list[str]) -> list[str]:
        tokens_no_punctuation = []

        for word in tokens:
            if word in list(string.punctuation):
                continue

            tokens_no_punctuation.append(word)

        return tokens_no_punctuation

    def _remove_digits(self, tokens: list[str]) -> list[str]:
        tokens_no_digits = []

        for word in tokens:
            if word in list(string.digits):
                continue

            tokens_no_digits.append(word)

        return tokens_no_digits
