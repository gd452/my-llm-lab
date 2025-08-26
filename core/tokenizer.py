"""
Day 5: Tokenizer Implementation
Character-level and BPE tokenizers for text processing
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


class CharacterTokenizer:
    """Character-level tokenizer - simplest form of tokenization"""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'  # Beginning of sequence
        self.eos_token = '<EOS>'  # End of sequence
    
    def fit(self, text: str):
        """Build vocabulary from text"""
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
        
        # Add unique characters
        unique_chars = sorted(set(text))
        for char in unique_chars:
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_id)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text to token ids"""
        ids = []
        
        if add_special_tokens:
            ids.append(self.char_to_id[self.bos_token])
        
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.char_to_id[self.unk_token])
        
        if add_special_tokens:
            ids.append(self.char_to_id[self.eos_token])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to text"""
        chars = []
        special_ids = {self.char_to_id[token] for token in 
                      [self.pad_token, self.unk_token, self.bos_token, self.eos_token]}
        
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
        
        return ''.join(chars)


class SimpleBPETokenizer:
    """Simplified Byte Pair Encoding tokenizer"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.word_tokenizer = re.compile(r'\w+|[^\w\s]|\s')
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
    
    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        """Get word frequencies from text"""
        words = self.word_tokenizer.findall(text.lower())
        word_freq = Counter(words)
        
        # Add word boundary markers
        word_freq_with_boundary = {}
        for word, freq in word_freq.items():
            # Add space to mark word boundaries
            word_with_boundary = ' '.join(list(word)) + ' </w>'
            word_freq_with_boundary[word_with_boundary] = freq
        
        return word_freq_with_boundary
    
    def _get_pair_frequencies(self, word_freq: Dict[str, int]) -> Counter:
        """Count frequency of adjacent pairs"""
        pair_freq = Counter()
        
        for word, freq in word_freq.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] += freq
        
        return pair_freq
    
    def _merge_pair(self, word_freq: Dict[str, int], pair: Tuple[str, str]) -> Dict[str, int]:
        """Merge most frequent pair in vocabulary"""
        new_word_freq = {}
        merged = pair[0] + pair[1]
        
        for word, freq in word_freq.items():
            tokens = word.split()
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            new_word = ' '.join(new_tokens)
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def fit(self, text: str, num_merges: int = 100):
        """Learn BPE merges from text"""
        # Initialize with character-level tokens
        word_freq = self._get_word_frequencies(text)
        
        # Learn merges
        for _ in range(num_merges):
            pair_freq = self._get_pair_frequencies(word_freq)
            if not pair_freq:
                break
            
            # Get most frequent pair
            most_frequent_pair = pair_freq.most_common(1)[0][0]
            self.merges.append(most_frequent_pair)
            
            # Merge the pair
            word_freq = self._merge_pair(word_freq, most_frequent_pair)
        
        # Build vocabulary
        self._build_vocabulary(word_freq)
        print(f"Learned {len(self.merges)} merges")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def _build_vocabulary(self, word_freq: Dict[str, int]):
        """Build vocabulary from word frequencies"""
        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.id_to_token[idx] = token
        
        # Add all unique tokens
        all_tokens = set()
        for word in word_freq.keys():
            tokens = word.split()
            all_tokens.update(tokens)
        
        for token in sorted(all_tokens):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Apply BPE to a single word"""
        # Add word boundary
        tokens = list(word) + ['</w>']
        
        # Apply merges
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token ids"""
        ids = []
        
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])
        
        # Tokenize by words first
        words = self.word_tokenizer.findall(text.lower())
        
        for word in words:
            if word.strip():  # Non-whitespace
                tokens = self._tokenize_word(word)
                for token in tokens:
                    if token in self.vocab:
                        ids.append(self.vocab[token])
                    else:
                        ids.append(self.vocab[self.unk_token])
            else:  # Whitespace
                if ' ' in self.vocab:
                    ids.append(self.vocab[' '])
        
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        tokens = []
        special_ids = {self.vocab[token] for token in 
                      [self.pad_token, self.unk_token, self.bos_token, self.eos_token]}
        
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            if id in self.id_to_token:
                token = self.id_to_token[id]
                tokens.append(token)
        
        # Join tokens and remove word boundaries
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()


class DataLoader:
    """Simple data loader for training"""
    
    def __init__(self, text: str, tokenizer, batch_size: int = 32, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        # Tokenize entire text
        self.token_ids = tokenizer.encode(text)
        self.num_tokens = len(self.token_ids)
        
        # Calculate number of batches
        self.num_batches = (self.num_tokens - 1) // (batch_size * seq_length)
    
    def get_batch(self, batch_idx: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Get a batch of data"""
        inputs = []
        targets = []
        
        for i in range(self.batch_size):
            # Calculate position in token sequence
            start_idx = batch_idx * self.batch_size * self.seq_length + i * self.seq_length
            
            if start_idx + self.seq_length + 1 > self.num_tokens:
                break
            
            # Input: current tokens
            input_seq = self.token_ids[start_idx:start_idx + self.seq_length]
            # Target: next tokens
            target_seq = self.token_ids[start_idx + 1:start_idx + self.seq_length + 1]
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return inputs, targets
    
    def __iter__(self):
        """Iterate through all batches"""
        for batch_idx in range(self.num_batches):
            yield self.get_batch(batch_idx)
    
    def __len__(self):
        return self.num_batches