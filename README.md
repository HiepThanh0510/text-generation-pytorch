# text-generation

### 1. Introduction 
In this repo, i implement simple text generation using different decoding methods with Transformers.

### 2. Install 
```
$ pip3 install -r requiresment.txt
```
### 3. Usage
```
$ git clone https://github.com/HiepThanh0510/text-generation && cd text-generation
```
+ Greedy Search (Beam Search with num_beam = 1)
    ```python
    $ python3 text_generation.py --generate_type greedy_search --max_new_tokens 20 --prompt "Quần đảo Hoàng Sa, Trường Sa là của" --no_repeat_ngram_size 3

    >  Việt Nam, nhưng Trung Quốc đã ngang ngược tuyên bố chủ quyền trên toàn bộ quần đảo này.
    ```

+ Beam Search: 
    ```python
    $ python3 text_generation.py --generate_type beam_search --max_new_tokens 42 --prompt "Việt Nam là quốc gia" --no_repeat_ngram_size 3 --num_beam 4
    
    >  đứng thứ 2 trên thế giới về sản xuất và xuất khẩu cà phê. Năm 2017, Việt Nam xuất khẩu khoảng 1,6 triệu tấn cà phê, tăng 13,4% so với năm 2016.
    ```
 