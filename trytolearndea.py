import numpy as np
import pandas as pd
from scipy.optimize import linprog
def ddf_dea(df, inputs, good_output, bad_output, rts="CRS"):
    X=df[inputs].to_numpy(dtype=float)
    '''
    X lấy các dòng trong bảng được coi là input chuyển thành
    dạng ma trận số nhờ câu lệnh to_numpy
    dtype=float--> coi những con số trong ma trận là số thực
    '''
    Y=df[good_output].to_numpy(dtype=float) # tương tự
    B=df[bad_output].to_numpy(dtype=float)
    n=X.shape[0]
    '''
    lệnh shape để xem diện tích của ma trận
    chả hạn có 3 ngân hàng 2 input thì ma trận X có 3 dòng và 2 hàng
    X.shape hay n sẽ là 1 ma trận [3,2]
    số dòng đc đánh số là o(vị trí bát đầu), số cột vị trí 1
    nên X,shape[0] là lấy só dòng hay là lấy số bank
    cũng có thể dùng Y.shape hoặc B.shape  
    '''  
    n_x=X.shape[1] #tìm số input
    n_y=Y.shape[1] #tìm số output tốt(good_output)
    n_b=B.shape[1] #tìm số output xấu(bad_output)
    n_vars=n+1
    '''
    Khi giải bài toán ddf-dea
    ta xây dựng fortainer dựa trên các ngân hàng trong đó
    mỗi ngân hàng có 1 landa để đưa ra xây dựng đường biên đó
    và hệ số β-beta xem ngân hàng đang đợt xét giảm input,tăng output tốt và giảm output xấu được tốt đa bao nhiêu beta%
    -->[λA​,λB​,λC​,β]=[0;0,7;0;8..]--> tổ hợp ngân hàng B và C
    '''
    betas=np.full(n,np.nan)
    #tạo ma trận n chỗ trống[NaN,NaN,...] để chứa beta của n ngân hàng
    
    #BẮT ĐẦU GIẢI TỪNG NGÂN HÀNG
    for o in range(n):
        x_o=X[o]
        y_o=Y[o]
        b_o=B[o]
        # o là ngân hàng đang được tính
        c = np.zeros(n_vars) 
        c[-1]= -1.0
        '''
        Tạo vecto có n_vars cột dạng [0;0;0.....0]
        còn có thể code là c = np.full(n_vars,0)
        c[-1] giá trị cuối của vector bằng -1
        do vecto cuối dành cho output xấu, đổi dấu với mục đích lm nó bé nhất có thể
        do ở đây chỉ có 1 output xấu nên chỉ sd c[-1]
        nếu nó là 2 output xấu thì là c[-1:-3]=-1.0
        '''
        A_ub=[]
        b_ub=[]
        for i in range(n_x):
            row = np.zeros(n_vars)
            row[:n] = X[:,i] # lấy tất cả các dòng còn i là cột
            # vd X[:,0]--> lấy toàn bộ dòng của cột assets
            '''
            row = np.zeros[n_vars] tạo ra vector n+1 số 0
            row[:n] từ vị trí đầu là 0 đến vị trí n-1 
            '''
            row[-1]=x_o[i]
            #row = [10, 5, 8, 8]-->10λA​+5λB​+8λC​+8β≤8
            A_ub.append(row)
            b_ub.append(x_o[i])
        for r in range(n_y):
            row = np.zeros(n_vars)
            row[:n]=-Y[:,r]
            row[-1]=y_o[r]
            A_ub.append(row)
            b_ub.append(-y_o[r])

        A_eq=[]
        b_eq=[]
        for k in range(n_b):
            row = np.zeros(n_vars)
            row[:n]=B[:,k]
            row[-1]=b_o[k]
            A_eq.append(row)
            b_eq.append(b_o[k])
            '''
            Nếu viết:Bλ <= bo​−βgb​
            thì ý nghĩa là:
            NPL của điểm tham chiếu có thể thấp hơn hoặc bằng mức 
            NPL sau cải thiện.
            Điều này cho phép nó "thấp hơn nữa".
            Trong một số mô hình DEA undesirable-output, 
            điều đó có thể là cách mô hình hóa hợp lệ. 
            Không có quy tắc chung rằng NPL bắt buộc phải dùng equality.
            '''
            bounds = [(0, None)] * n_vars
            '''
            bounds--> giới hạn.
            công thức (lower bound,upper_bound)
            do xây từ các trọng số từ các ngân hàng nên ko âm được
            '''
        #đưa máy giải các điều kiện
        res=linprog(
            c,
            A_ub=np.array(A_ub),#biến list thành mảng numpy
            b_ub=np.array(b_ub),
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method='highs'
            )
        if res.success:
            betas[o]=res.x[-1]
    out=df.copy()
    out["beta"]=betas
    out["độ hiệu quả"]=1-out["beta"]
    # tạo  bảng copy, betas là tiến bao nhiêu phần trăm sẽ đến biên
    #nên độ hiệu quả là 1-betas
    return out

# ROBUST
def robust(df, inputs, good_output, bad_output,npl_col="npl",
           x=0.1,n_sim=200,rts="crs",seed=123):
        # hàm viết trong trường hợp có nhiều hơn 1 bad outputs
        '''
        mức bất định bằng 0.1
        -> mô phỏng dữ liệu chạy trong khoảng dữ liệu x (1+/- 0.1)
        n_sim=200-->tạo bộ dữ liệu mới 200 lần rồi chạy
        seed=123--> tạo các dữ liệu ngẫu nhiên cố định nếu ko mỗi lần kq sẽ ra khác nhau
        '''
        rng = np.random.default_rng(seed)
        #np.random--> dùng để tạo các số ngẫu nhiên
        #default_rng tạo bộ ngẫu nhiên riêng khiến các code khác khi random ko bị ảnh hưởng bởi sự ngẫu nhiên trước đó
        goc=ddf_dea(df,inputs, good_output, bad_output, rts=rts)
        beta_goc = goc["beta"].to_numpy()
        n=len(df)
        npl_goc=df[npl_col].to_numpy(dtype=float)
        sims = np.full((n_sim,n),np.nan)
        '''
        tạo ra ma trận Nan
        với n_sim dòng và n cột--> mỗi lần chạy ra 1 cụm beta
        sẽ cho vào bảng này
        '''
        for s in range(n_sim):
             he_so_nhieu = rng.uniform(1-x,1+x,size=n)
             #tạo ra n số ngẫu nhiên trong khoảng bất định
             df_sim=df.copy()
             df_sim[npl_col]= npl_goc* he_so_nhieu
             ket_qua_sim = ddf_dea(df_sim,inputs, good_output, bad_output, rts=rts)
             sims[s,:]=ket_qua_sim["beta"].to_numpy()
             #cột 0 chứa bộ beta số lần đó
        out=df.copy()
        out["beta gốc"] = beta_goc
        out["beta_TB"] = np.nanmean(sims,axis=0)
        #np.nanmean() tính giá trị trung bình, bỏ qua NaN
        '''
        axis=0-->Tính theo chiều dọc, 
        tức là lấy 200 simulation của từng bank 
        để tính trung bình.
        '''
        out["độ lệch chuẩn"] = np.nanstd(sims,axis=0)
        out["beta_p95"] = np.nanpercentile(sims,95,axis = 0)
        #sd phân vị ở mức 95% để đánh giá thận trọng
        # p95 cua beta = kich ban XAU nhat (beta cao = kem hieu qua) trong 95% kich ban
        out["beta_robust"] = 1 - out["beta_p95"]
        return out


# NẠP DỮ LIỆU ĐẦU VÀO
def load_data():
     return pd.read_csv(r"C:\Users\DELL\Documents\DEA\Bo_Du_Lieu_Ban_Dau.csv")
if __name__ == "__main__":
    INPUT_COLS = ["total_assets", "opex"]
    GOOD_OUTPUT_COLS = ["total_loans"]
    BAD_OUTPUT_COLS = ["npl"]
    df = load_data()
    result = ddf_dea(df,INPUT_COLS,GOOD_OUTPUT_COLS,BAD_OUTPUT_COLS,rts="crs")
    print(result[["bank", "year", "beta", "độ hiệu quả"]]
    .head(10)#lấy 10 dòng đầu của datafame
    .round(4)#làm tròn đến 4 chứ số thập phân
    .to_string(index=False)#index=false là không ghi thứ tự dòng in
    )
    print(f"\n Tổng số ngân hàng:{len(result)}")
    print(f"số ngân hàng hiệu quả 100%: {(result['beta'] < 1e-6).sum()} / {len(result)}")
    '''
    với beta<1e-6 thì trả kết quả true
    true=1--> sum() tính tổng tất cả các dòng thành số ngân hàng hiệu quả
    '''
    print("\n--- Hieu qua trung binh theo nam ---")
    print(result.groupby("year")["độ hiệu quả"].mean().round(4).to_string())
    #tính trung bình hiệu quả theo năm
    result.to_excel("ddf_dea_crs_pooled_results.xlsx", index=False)
    # index=false, không lưu dòng đầu tiên mặc định 0,1,2,...
    print("\nDa luu ket qua vao ddf_dea_crs_pooled_results.xlsx")
    result_1 = robust(df, INPUT_COLS,GOOD_OUTPUT_COLS,BAD_OUTPUT_COLS,npl_col="npl",
           x=0.1,n_sim=200,rts="crs",seed=123)
    print()
    print(result_1[["bank", "year", "beta gốc", "beta_TB", "độ lệch chuẩn", "beta_p95", "beta_robust"]]
    .head(10)#lấy 10 dòng đầu của datafame
    .round(4)#làm tròn đến 4 chứ số thập phân
    .to_string(index=False)#index=false là không ghi thứ tự dòng in
    )
    print(f"\n Tổng số ngân hàng:{len(result)}")
    print(f"số ngân hàng hiệu quả 100%: {(result['beta'] < 1e-6).sum()} / {len(result)}")
    print("\n--- Hieu qua trung binh theo nam ---")
    print(result.groupby("year")["độ hiệu quả"].mean().round(4).to_string())
    result.to_excel("ddf_dea_robust.xlsx", index=False)
    # index=false, không lưu dòng đầu tiên mặc định 0,1,2,...
    print("\nDa luu ket qua vao ddf_dea_robust.xlsx.xlsx")