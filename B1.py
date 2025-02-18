import cv2
#doc anh
img = cv2.imread("E:\OneDrive - MSFT\Pictures\Screenshots\Screenshot (7).png",1)
#xuat anh
cv2.imshow("anh",img)
k = cv2.waitKey()