with open("output.test_2018_flickr.ens7.normal.beam12") as textfile1, open("index.txt") as textfile2:
    with open("UMONS_1_FLICKR_DE_DeepGru_C","w+") as textfile3:
        for x, y in zip(textfile1, textfile2):
            x = x.strip()
            y = y.strip()
            textfile3.write("DeepGru\t"+y+"\t"+x+"\t1\tC\n")
