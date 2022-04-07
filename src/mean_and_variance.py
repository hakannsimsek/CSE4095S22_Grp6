list_of_dict_values = list(word_map.items())
statisticOfWordPairs = {}

def my_function(key, pair):
    arr = [0, 0, 0, 0, 0]
    for i in range(-2, 3):
        if i == 0:
            continue
        adjacents = key[1][i]
        if pair in adjacents.keys():
            payir = adjacents[pair]
            arr[i+2] = adjacents[pair]

    #print("key : " ,key )
    #print("pair" , pair)
    mean,count = calculateMean(arr)
    #print("mean : ", mean)
    #if mean % 1 != 0:
        #print("")
    variance = calculateVariance(arr,mean)
    #print("var : ", variance)
    stdDev = math.sqrt(variance)
    #print("std dev : ", stdDev)
    #print("\n")
    return mean,stdDev,arr,count

def calculateVariance(arr,mean):
    sum = 0
    count = 0
    for i in range(0,5):
        count += arr[i]
        sum += arr[i]*( ((i-2) - mean)*((i-2) - mean) )

    if count < 2 :
        return 0
    sSqaure = (sum/(count-1))
    #print("")
    #if sSqaure==-1:
        #print("")

    return sSqaure


def calculateMean(arr):
    sum = 0
    count = 0
    for i in range(0,5):
        count += arr[i]
        sum += arr[i] * (i-2)

    if count <= 1 :
        return 0,count
    mean = sum/count
    return mean,count


def mean_and_variance(list_of_dict_values):
    countSum = 0
    numberOFPairs = 0
    for key in list_of_dict_values:
            for i in range(-2, 3):
                #print("here : " , i, key[0], key[1][i])
                #if key[0] == "hem":
                adjacents = key[1][i]
                if i == 0:
                    continue
                #if key[0] == "davacı":
                for pair in adjacents:
                    ##print("key zero ", key[0])
                    mean,stdDev,arr,count = my_function(key, pair)
                    if count > 8 and mean > 0.8 and mean < 1.2 and stdDev < 0.3:
                        print("\"",key[0],"\"\"", pair,"\"", " mean : ", mean, " std dev : ", stdDev , arr ,count)
                        countSum += count
                        numberOFPairs += 1
                        #print("countSum : ", countSum, " #pairs : ", numberOFPairs)


                    #print("key zero ", key[0])
                    if key[0] not in statisticOfWordPairs.keys():
                        statisticOfWordPairs[key[0]] = []
                    statisticOfWordPairs[key[0]].append({
                        0: pair,
                        1: mean,
                        2: stdDev,
                        3: arr
                    })

                    #print(i, pair, adjacents[pair])

# print("avg count : ", countSum/numberOFPairs)
# print("")