with open('Viet74K.txt', 'r', encoding='utf-8') as infile, open('vi_VN.dic', 'w', encoding='utf-8') as outfile:
    outfile.write(str(sum(1 for _ in infile)) + '\n')  # Ghi số lượng từ
    infile.seek(0)
    for line in infile:
        word, freq = line.strip().split(',')
        outfile.write(f'{word}/{freq}\n')