# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Temperature càng cao thì độ sáng tạo trong câu phản hồi càng nhiều*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Chọn 0.1->0.3 vì hỗ trợ khách hàng cần độ chính xác cao, ngưỡng này vẫn đảm bảo được độ chính xác mà không khiến câu trả lời quá máy móc*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *GPT-4o đắt gấp 17 lần*
> *Nên dùng GPT-4o khi tác vụ đòi hỏi suy luận phức tạp như sinh code, xử lý ngữ cảnh dài. Trường hợp dùng mini khi tác vụ yêu cầu không quá phức tạp*
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Cùng câu hỏi nhưng hai persona cho ra
> hai văn phong khác hẳn. **Giáo viên tiểu học**: dùng ẩn dụ đời thường ("Sổ
> tay thần kỳ", trò chơi đổi nhãn vở), xưng "thầy/cô–con", câu ngắn, gần như
> KHÔNG thuật ngữ (316 từ / 488 token, ~10,3s). **Chuyên gia tài chính**: dày
> đặc thuật ngữ tiếng Anh (Distributed Ledger Technology, Append-only,
> Double-spending, SHA-256, Merkle Tree, Avalanche Effect), chia mục học thuật
> (286 từ / 426 token, ~8,6s). Thú vị là độ dài xấp xỉ nhau, nhưng bản B ít từ
> hơn mà nhiều ký tự hơn vì từ chuyên môn dài. Tức là system prompt gần như
> không đổi "sự thật" cốt lõi mà chi phối mạnh vốn từ, ví dụ và văn phong
> (register) — nó định hình model ĐÓNG VAI ai chứ không phải model BIẾT gì.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> (Đo thật trên đoạn văn 115 từ.) `count_tokens` bằng tiktoken (o200k_base của
> gpt-4o) = **149 token**; ước lượng `số_từ / 0.75` = **153,3 token**. Hai con
> số chỉ chênh **~2,9%** (ước lượng nhỉnh hơn chút), tức gần như bằng nhau,
> khoảng **1,30 token/từ**. Đáng chú ý: vì lab dùng model Gemini nên
> `count_tokens` rơi vào nhánh dự phòng `char//4` = **127 token**, THẤP hơntiktoken thật → ước tính chi phí theo fallback hơi hụt cho tiếng Việt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với phản hồi DÀI và tương tác thời gian thực —
> chatbot, trợ lý viết code/văn bản — nơi bắt người dùng chờ trọn câu trả lời
> sẽ thấy "đơ". Đo thực tế trong lab: một phản hồi đầy đủ của gemini-flash-latest
> mất ~7–10s; nếu stream thì chữ đầu tiên hiện sau ~1–2s nên cảm giác nhanh hơn
> hẳn dù tổng thời gian không đổi, và người dùng có thể đọc dần hoặc hủy sớm.
> Ngược lại, non-streaming phù hợp khi output ngắn, hoặc khi cần TRỌN kết quả
> trước khi xử lý tiếp — parse JSON, gọi hàm/tool, tính toán, kiểm duyệt nội
> dung — vì lúc đó nhận từng mảnh chẳng lợi mà còn làm code phức tạp hơn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> `retry_with_backoff` với `base_delay=0.1` và hỏng 3 lần đầu chờ
> đúng 0.1 + 0.2 + 0.4 ≈ **0,70s** (delay = base_delay·2^attempt) trước khi
> thành công ở lần gọi thứ 4. So với delay cố định, exponential backoff giãn
> dần khoảng chờ, nên khi API quá tải, client lùi lại ngày càng xa, cho server
> thời gian hồi phục thay vì dội liên tục. Nếu hàng nghìn client cùng retry với
> CÙNG một delay cố định, chúng sẽ gọi lại đúng cùng thời điểm (thundering
> herd), tạo các đợt sóng tải trùng phách khiến API càng quá tải — vì vậy thực
> tế còn thêm "jitter" (ngẫu nhiên hoá delay) để rải đều các lần retry.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Chọn persona là phân tích hợp đồng bảo hiểm. System promt:"Bạn là nhân viên tư vấn bảo hiểm với nhiều năm kinh nghiệm, hãy đọc yêu cầu của user thông qua {user_msg} và tìm kiếm các tài liệu trên mạng để tư vấn cho khách. Các luật phải tuân theo: 1. Nếu không biết thì phải nói là không biết 2. Tuyệt đối không bịa ra các chi phí lung tung trong hợp đồng 3.Các điều khoản hợp đồng phải được trích xuất từ nguồn*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: bộ nhớ hội thoại chỉ giữ 3 lượt cuối (`history =
> history[-6:]`) — trợ lý QUÊN mọi thứ trước đó, nên nếu người dùng nhắc lại
> thông tin đã nói ở đầu phiên thì nó không nhớ. Ngoài ra không có bộ nhớ dài
> hạn giữa các phiên, không kiểm duyệt nội dung, và `count_tokens` dùng fallback
> nên thống kê token/chi phí hơi lệch (thấp hơn thực tế cho tiếng Việt).
>
> Cải thiện cụ thể: thay "cắt cứng" bằng **cửa sổ trượt + tóm tắt** (sliding
> window + running summary). Triển khai: khi `history` vượt N lượt, gọi model
> tóm tắt các lượt cũ thành một đoạn "summary" ngắn và nhét vào một message
> `role="system"` phụ, đồng thời vẫn giữ nguyên 3 lượt gần nhất ở dạng chi tiết.
> Nhờ vậy trợ lý nhớ được bối cảnh xa mà token mỗi lượt không phình to.


---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
