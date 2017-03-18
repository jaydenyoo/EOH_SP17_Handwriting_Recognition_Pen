"""
This python code uses collected dataset of each Alphabet and generate a model.
"""

import xlrd, operator, math
import numpy as np

sheets = []
for i in range(1, 27):
    sheets.append(xlrd.open_workbook('26/'+str(i)+'.xlsx').sheet_by_name('Simple Data'))


#Sample Data Collection
CompleteList = []
CompleteMeanList = []
CompleteSampleList = []
firstdump = 0
samplerowsheet = 20
samplerow = samplerowsheet - firstdump
index = 1
for sheetIter in sheets:
    # print("Index is ", index)
    Alpha = []
    k = 1
    i = 0
    while(k < sheetIter.nrows-1):
        subAlpha = []
        for i in range(firstdump, samplerowsheet):
            sub2Alpha = []
            if (index == 7):
                for j in range(1, 7):
                    sub2Alpha.append(sheetIter.cell(i+k, j).value)
                subAlpha.append(sub2Alpha)
            else:
                for j in range(1, 7):
                    # print(index, i+k, j)
                    sub2Alpha.append(sheetIter.cell(i+k, j).value)
                subAlpha.append(sub2Alpha)
        Alpha.append(subAlpha)
        k = i+k+(23-samplerowsheet)

    CompleteSampleList.append(Alpha)
    arr = []
    for j in range(0, samplerow):
        meanlist = []
        for k in range(0, 6):
            x = []
            for i in range(0, len(Alpha)):
                # for j in range(0, 21):
                # print((i,j,k))
                x.append(Alpha[i][j][k])
            meanlist.append(np.mean(x))
        arr.append(meanlist)

    # fourD = []
    # for i in range(0, 21):
    #     threeD = []
    #     for j in range(0, 6):
    #         twoD = []
    #         for u in range(0, 21):
    #             oneD = []
    #             for v in range(0, 6):
    #                 temp = 0
    #                 for s in range(0, len(Alpha)):
    #                     # print(((i,j,u,v),s))
    #                     # temp += (Alpha[s][i][j]) - arr[i][j])*(Alpha[s][u][v] - arr[u][v])
    #                     temp1 = Alpha[s][i][j]
    #                     temp2 = arr[i][j]
    #                     temp3 = Alpha[s][u][v]
    #                     temp4 = arr[u][v]
    #                     temp = (temp1 - temp2)*(temp3 - temp4)
    #                 oneD.append(temp*(1/((len(Alpha)**(1/2)))))
    #             twoD.append(oneD)
    #         threeD.append(twoD)
    #     fourD.append(threeD)


    #
    #
    # CompleteList.append(fourD)
    CompleteMeanList.append(arr)
    index += 1

#CoVariace Calculation
var = []
for k in range(26):
    templistlist = []
    for i in range(samplerow):
        templist = []
        for j in range(6):
            temp = 0
            for s in range(len(CompleteSampleList[k])):
                temp += (CompleteSampleList[k][s][i][j] - CompleteMeanList[k][i][j])**2
            templist.append(temp*(1/len(CompleteSampleList[k])))
        templistlist.append(templist)
    var.append(templistlist)


# print("First Part Done")

##########################################################
# Option 4 Original Error Counting Method
# CompleteErrorList = []
# index = 1
# for var in range(0, 26):
#     TempErrorList = []
#     for sample in CompleteSampleList[1]:
#         print(index)
#         # print(sample)
#         index += 1
#         ErrorList = []
#         for alphabet in range(0, 26):
#             temp = 0
#             for i in range(0, 21):
#                 for j in range(0, 6):
#                     for u in range(0, 21):
#                         for v in range(0, 6):
#                             temp += (sample[i][j]*sample[u][v] - (CompleteMeanList[alphabet][i][j] * CompleteMeanList[alphabet][u][v]))**2
#             ErrorList.append(temp)
#         TempErrorList.append(ErrorList.index(min(ErrorList)))
#     CompleteErrorList.append(TempErrorList)
#
# for n in range(len(CompleteErrorList)):
#     print(CompleteErrorList[n])
############################################################

mulist = [[[] for i in range(6)] for j in range(samplerow)]

for i in range(0, samplerow):
    for j in range(0, 6):
        for abc in range(0, 26):
            mulist[i][j].append([abc, CompleteMeanList[abc][i][j]])
        mulist[i][j].sort(key=operator.itemgetter(1))

# for i in range(21):
#     for j in range(6):
#         for abc in range(0, 26):
#             if (mulist[i][j][abc][0] == 2):
#                 if ( 0 < abc < 25):
#                     print((mulist[i][j][abc-1][0], mulist[i][j][abc][0], mulist[i][j][abc+1][0]),(mulist[i][j][abc-1][1], mulist[i][j][abc][1], mulist[i][j][abc+1][1]), "IN")
#                 elif (abc == 0):
#                     print((mulist[i][j][abc][0], mulist[i][j][abc+1][0]), (mulist[i][j][abc][1], mulist[i][j][abc+1][1]), "BEG")
#                 elif (abc == 25):
#                     print((mulist[i][j][abc-1][0], mulist[i][j][abc][0]), (mulist[i][j][abc-1][1], mulist[i][j][abc][1]), "END")


Eta = []
for i in range(0, samplerow):
    templistlist = []
    for j in range(0, 6):
        templist = []
        for muidx in range(0, len(mulist[i][j])-1):
            k1 = mulist[i][j][muidx][0]
            k2 = mulist[i][j][muidx+1][0]
            TempErrorList = []
            found = 0
            for idx in range(int(mulist[i][j][muidx][1]), int(mulist[i][j][muidx+1][1])):
                temp1 = (1/(math.sqrt(2*math.pi*var[k1][i][j])))*math.exp((math.pow(idx-CompleteMeanList[k1][i][j], 2))*(-0.5*(var[k1][i][j])))
                temp2 = (1/(math.sqrt(2*math.pi*var[k2][i][j])))*math.exp((math.pow(idx-CompleteMeanList[k2][i][j], 2))*(-0.5*(var[k2][i][j])))

                if temp2 > temp1:
                    templist.append([k1, k2, idx])
                    found = 1
                    break
            if (found == 0):
                if (muidx == 0):
                    templist.append([k1, k2, int((mulist[i][j][muidx][1]+mulist[i][j][muidx+1][1])/2)])
                else:
                    templist.append([k1, k2, int((mulist[i][j][muidx+1][1] + max(mulist[i][j][muidx][1], templist[-1][-1]))/2)])

        templistlist.append(templist)
    Eta.append(templistlist)
#
# etatxt = open("etalist.txt", "w")
# for i in range(21):
#     for j in range(6):
#         for k in range(len(Eta[i][j])):
#             etatxt.write(Eta[i][j][k])
#         etatxt.write("\n")
print(Eta)

##########################################################
#Eta Indexing Error Check
# for i in range(len(Eta)):
#     for j in range(6):
#         print("------------------------------")
#         for k in range(25-1):
#             # print ((Eta[i][j][k][0], Eta[i][j][k][1]))
#             if (Eta[i][j][k][1] != Eta[i][j][k+1][0]):
#                 print("ERROR", (i,j,k))
###########################################################

###########################################################
# Output Average Interval of Eta for each Alphabet
# cnt = [0 for i in range(26)]
# plus = [0 for i in range(26)]
# intcnt = []
# for i in range(21):
#     for j in range(6):
    # for k in range(len(Eta[i][j])):
    #     if (Eta[i][j][k][1] == 2):
    #         # if (k == len(Eta[i][j])-1):
    #         #     # print(Eta[i][j][k][2], "MAX")
    #         #     print((Eta[i][j][k-1][2],Eta[i][j][k][2]))
    #         # elif (k == 0):
    #         #     print((Eta[i][j][k][2],Eta[i][j][k+1][2]))
    #         # else:
    #             # print(abs(Eta[i][j][k][2] - Eta[i][j][k+1][2]), "IN")
    #             print((Eta[i][j][k][0], Eta[i][j][k][1], Eta[i][j][k+1][1], Eta[i][j][k][2], Eta[i][j][k+1][2]),"\n")
    #             # print("-->", (CompleteMeanList[Eta[i][j][k][0]][i][j], CompleteMeanList[Eta[i][j][k][1]][i][j], CompleteMeanList[Eta[i][j][k+1][1]][i][j]),"\n")
    #             # plus[Eta[i][j][k][1]] += abs(Eta[i][j][k][2] - Eta[i][j][k+1][2])
    #             # cnt[Eta[i][j][k][1]] += 1

# print("MEAN")
# for i in mulist[0][0]:
#     print(i)
# print("ETA")
# for i in ((Eta[0][0])):
#     print(i)


#
# for i in range(26):
#     if (cnt[i] == 0):
#         intcnt.append(0)
#     else:
#         intcnt.append(plus[i]/cnt[i])
#
# print(intcnt)
##########################################################


##########################################################
# Count Detection
#
errorcount = []

for m in range(26):
    # cnt = [0 for a in range(len(CompleteSampleList[m]))]
    count = [[0 for j in range(26)] for k in range(len(CompleteSampleList[m]))]
    for k in range(0, len(CompleteSampleList[m])):
        for i in range(samplerow):
            for j in range(6):
                for etaidx in range(0, len(Eta[i][j])-1):
                    # print((Eta[i][j][etaidx][1], Eta[i][j][etaidx+1][0]))
                    if (etaidx == 0 and CompleteSampleList[m][k][i][j] <= Eta[i][j][etaidx][2]):
                        count[k][Eta[i][j][etaidx][0]] += 1
                        break
                    elif (etaidx == len(Eta[i][j])-1 and CompleteSampleList[m][k][i][j] > Eta[i][j][etaidx][2]):
                        count[k][Eta[i][j][etaidx][1]] += 1
                        break
                    elif (Eta[i][j][etaidx][2] < CompleteSampleList[m][k][i][j] <= Eta[i][j][etaidx+1][2]):
                        count[k][Eta[i][j][etaidx][1]] += 1
                        break
    temp = 0
    for b in range(0, len(CompleteSampleList[m])):
        if (count[b].index(max(count[b])) == m):
            temp += 1
    errorcount.append(temp)
print(errorcount)

j = 0
for i in errorcount:
    j += i
print(j/(48*26))

# print(len(CompleteSampleList[6]))
    # result = []
    #
    # for i in count:
    #     result.append(i.index(max(i)))

# for i in range(samplerow):
#     for j in range(6):
#         for k in range(25):
#             for l in range(3):
#                 print(Eta[i][j][k][l])

# print(result)
#######################################################
