from core.utils.logger import logger

"""
Generate synthetic Vietnamese legal cases for LegalGraphRAG.

Covers 3 legal domains:
  1. Luật Lao động (Labor Law)
  2. Bộ luật Dân sự (Civil Code)
  3. Luật Bảo hiểm xã hội (Social Insurance Law)

Output: data/processed/cases_with_feature.json
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LAO_DONG_CASES = [
    {
        "name": ["Nguyễn Văn Minh"],
        "fact": "Anh Nguyễn Văn Minh làm việc tại Công ty TNHH ABC từ tháng 01/2019 theo hợp đồng lao động có thời hạn 2 năm. Đến tháng 06/2020, công ty ra quyết định sa thải anh với lý do 'vi phạm kỷ luật lao động' nhưng không tổ chức họp hội đồng kỷ luật, không thông báo trước 30 ngày và không chi trả trợ cấp thôi việc. Anh Minh cho rằng việc sa thải là trái pháp luật và yêu cầu công ty bồi thường, tái sử dụng lao động hoặc trả trợ cấp.",
        "dispute": ["Sa thải trái pháp luật", "Không chi trả trợ cấp thôi việc"],
        "law": ["36", "38", "40", "41"],
        "features": {
            "parties_info": ["người sử dụng lao động", "công ty TNHH"],
            "dispute_acts": [
                "sa thải không đúng thủ tục",
                "không họp hội đồng kỷ luật",
                "không thông báo trước",
            ],
            "subject_matter": ["mất việc làm", "thiệt hại thu nhập"],
            "fault_and_evidence": ["cố ý đơn phương chấm dứt hợp đồng"],
        },
    },
    {
        "name": ["Trần Thị Hoa"],
        "fact": "Chị Trần Thị Hoa làm nhân viên kế toán tại Công ty Cổ phần XYZ từ năm 2017. Từ tháng 3/2021, công ty liên tục trả lương chậm từ 15-30 ngày, trả thiếu phụ cấp ăn trưa và không thanh toán tiền làm thêm giờ trong suốt 6 tháng. Tổng số tiền bị nợ lương và phụ cấp ước tính 42 triệu đồng. Chị Hoa đã khiếu nại nội bộ nhưng không được giải quyết, buộc phải khởi kiện ra Tòa án nhân dân.",
        "dispute": ["Nợ lương người lao động", "Không thanh toán làm thêm giờ"],
        "law": ["90", "97", "98", "107"],
        "features": {
            "parties_info": ["người sử dụng lao động", "công ty cổ phần"],
            "dispute_acts": [
                "trả lương chậm",
                "trả thiếu phụ cấp",
                "không thanh toán làm thêm giờ",
            ],
            "subject_matter": ["nợ lương 42 triệu đồng", "thiệt hại tài chính"],
            "fault_and_evidence": ["cố ý chiếm dụng tiền lương"],
        },
    },
    {
        "name": ["Lê Quốc Hùng"],
        "fact": "Anh Lê Quốc Hùng ký hợp đồng lao động không xác định thời hạn với Xí nghiệp Xây dựng Bình Dương. Ngày 15/8/2022, trong khi thi công tại công trình, anh bị tai nạn lao động gãy tay phải do giàn giáo không đảm bảo an toàn. Xí nghiệp không mua bảo hiểm tai nạn lao động cho anh và từ chối bồi thường với lý do anh tự ý vi phạm quy trình an toàn. Anh Hùng yêu cầu bồi thường thiệt hại và chi phí điều trị tổng cộng 180 triệu đồng.",
        "dispute": [
            "Tai nạn lao động do lỗi người sử dụng lao động",
            "Không mua bảo hiểm tai nạn lao động",
        ],
        "law": ["130", "131", "145", "146"],
        "features": {
            "parties_info": ["người sử dụng lao động", "xí nghiệp xây dựng"],
            "dispute_acts": [
                "giàn giáo không an toàn",
                "không mua BHXH tai nạn",
                "từ chối bồi thường",
            ],
            "subject_matter": [
                "gãy tay phải",
                "chi phí y tế 180 triệu",
                "mất khả năng lao động tạm thời",
            ],
            "fault_and_evidence": ["sơ suất nghiêm trọng về an toàn lao động"],
        },
    },
    {
        "name": ["Phạm Thị Lan"],
        "fact": "Chị Phạm Thị Lan đang mang thai tháng thứ 7 và làm việc tại nhà máy may Đồng Nai. Tháng 10/2023, quản lý yêu cầu chị làm thêm giờ vào ban đêm và làm việc trong điều kiện nặng nhọc trái quy định. Khi chị từ chối với lý do sức khỏe thai sản, công ty đã chuyển chị sang bộ phận khác với mức lương thấp hơn và không thanh toán chế độ thai sản đúng quy định. Chị Lan khởi kiện yêu cầu phục hồi vị trí và thanh toán đủ chế độ thai sản 6 tháng.",
        "dispute": ["Vi phạm quyền lao động nữ mang thai", "Không thanh toán chế độ thai sản"],
        "law": ["137", "138", "139"],
        "features": {
            "parties_info": ["người sử dụng lao động", "nhà máy may"],
            "dispute_acts": [
                "bắt làm thêm giờ ban đêm khi mang thai",
                "chuyển vị trí trái quy định",
                "thiếu chế độ thai sản",
            ],
            "subject_matter": ["mất thu nhập thai sản", "ảnh hưởng sức khỏe bà mẹ và thai nhi"],
            "fault_and_evidence": ["phân biệt đối xử lao động nữ mang thai"],
        },
    },
    {
        "name": ["Vũ Đình Thắng"],
        "fact": "Anh Vũ Đình Thắng làm giám đốc kinh doanh tại Tập đoàn TechViet từ 2018 đến 2023. Khi nghỉ việc, công ty kiện anh vi phạm điều khoản không cạnh tranh 12 tháng vì gia nhập công ty đối thủ sau 3 tháng. Anh Thắng lập luận rằng điều khoản không hợp lệ vì công ty không trả khoản bù đắp theo thỏa thuận. Tranh chấp liên quan đến hợp đồng lao động và bảo mật thương mại.",
        "dispute": ["Tranh chấp điều khoản không cạnh tranh", "Vi phạm bảo mật thương mại"],
        "law": ["21", "23"],
        "features": {
            "parties_info": ["người lao động cấp cao", "giám đốc kinh doanh"],
            "dispute_acts": [
                "gia nhập công ty đối thủ trong thời hạn cấm",
                "tiết lộ thông tin thương mại",
            ],
            "subject_matter": ["thiệt hại thương mại", "mất lợi thế cạnh tranh"],
            "fault_and_evidence": ["vi phạm điều khoản hợp đồng"],
        },
    },
    {
        "name": ["Ngô Thị Bích"],
        "fact": "Chị Ngô Thị Bích là công nhân may tại Khu công nghiệp Bình Dương, tham gia đình công tập thể ngày 20/5/2022 do tranh chấp về tiền lương và điều kiện làm việc. Sau đình công, công ty sa thải 12 công nhân tham gia đình công, trong đó có chị Bích. Chị và đại diện công đoàn cho rằng đình công là hợp pháp theo Bộ luật Lao động và việc sa thải là trả thù người lao động.",
        "dispute": ["Sa thải người tham gia đình công hợp pháp", "Trả thù người lao động"],
        "law": ["198", "204", "217"],
        "features": {
            "parties_info": ["người sử dụng lao động", "công ty may mặc"],
            "dispute_acts": [
                "sa thải sau đình công",
                "trả thù người lao động",
                "vi phạm quyền đình công",
            ],
            "subject_matter": ["mất việc làm", "thiệt hại thu nhập"],
            "fault_and_evidence": ["cố ý trả thù người lao động hợp pháp"],
        },
    },
    {
        "name": ["Đinh Văn Tú"],
        "fact": "Anh Đinh Văn Tú làm việc tại Công ty Vận tải Bắc Nam theo hợp đồng thử việc 3 tháng với lương thử việc bằng 70% lương chính thức. Sau 3 tháng, công ty không ký hợp đồng chính thức mà gia hạn thử việc thêm 3 tháng nữa. Anh Tú yêu cầu ký hợp đồng lao động chính thức vì thời hạn thử việc tối đa theo luật đã hết. Công ty từ chối và chấm dứt hợp đồng thử việc.",
        "dispute": ["Vi phạm quy định thử việc", "Chấm dứt hợp đồng trái pháp luật"],
        "law": ["25", "26", "27"],
        "features": {
            "parties_info": ["người sử dụng lao động", "công ty vận tải"],
            "dispute_acts": [
                "kéo dài thử việc quá hạn",
                "không ký hợp đồng chính thức",
                "chấm dứt hợp đồng không lý do",
            ],
            "subject_matter": ["thiệt hại thu nhập 30% lương", "mất cơ hội việc làm"],
            "fault_and_evidence": ["lách luật về thử việc"],
        },
    },
    {
        "name": ["Hoàng Thị Mai"],
        "fact": "Chị Hoàng Thị Mai làm kế toán trưởng tại Công ty TNHH Dệt May Hà Nội với mức lương thỏa thuận 25 triệu/tháng. Từ tháng 1/2023, công ty tự ý cắt giảm lương xuống 18 triệu với lý do khó khăn kinh doanh mà không có sự đồng ý của chị, không thông báo trước và không ký phụ lục hợp đồng. Chị Hoàng khởi kiện yêu cầu trả đủ lương theo hợp đồng gốc và bồi thường phần lương bị cắt 6 tháng.",
        "dispute": ["Tự ý cắt giảm lương không thỏa thuận", "Vi phạm hợp đồng lao động"],
        "law": ["28", "33", "90"],
        "features": {
            "parties_info": ["người sử dụng lao động", "công ty dệt may"],
            "dispute_acts": [
                "tự ý giảm lương",
                "không ký phụ lục hợp đồng",
                "vi phạm điều khoản lương",
            ],
            "subject_matter": ["nợ lương 42 triệu đồng (7 triệu x 6 tháng)", "thiệt hại tài chính"],
            "fault_and_evidence": ["cố ý vi phạm điều khoản hợp đồng"],
        },
    },
]

DAN_SU_CASES = [
    {
        "name": ["Trần Văn Bình"],
        "fact": "Anh Trần Văn Bình ký hợp đồng vay 500 triệu đồng với chị Nguyễn Thị Thu, lãi suất 2%/tháng, thời hạn 12 tháng. Sau 8 tháng, anh Bình ngừng trả lãi và không trả gốc. Chị Thu khởi kiện yêu cầu thu hồi gốc 500 triệu, lãi theo hợp đồng và lãi chậm trả. Anh Bình phản tố lãi suất 2%/tháng vượt quá mức trần 20%/năm theo Bộ luật Dân sự, đề nghị tòa tuyên hợp đồng vô hiệu một phần về lãi suất.",
        "dispute": ["Tranh chấp hợp đồng vay tài sản", "Tranh chấp lãi suất vượt trần"],
        "law": ["463", "466", "468"],
        "features": {
            "parties_info": ["người vay tiền", "cá nhân"],
            "dispute_acts": ["không trả nợ gốc", "ngừng trả lãi", "thỏa thuận lãi suất vượt trần"],
            "subject_matter": ["gốc 500 triệu đồng", "lãi phát sinh 8 tháng"],
            "fault_and_evidence": ["cố tình không trả nợ"],
        },
    },
    {
        "name": ["Lê Thị Hương"],
        "fact": "Chị Lê Thị Hương mua căn hộ chung cư của Công ty BĐS Thịnh Vượng theo hợp đồng ký tháng 3/2021, giá 2,5 tỷ đồng, cam kết bàn giao tháng 12/2022. Đến tháng 6/2023 chủ đầu tư vẫn chưa bàn giao, từ chối hoàn tiền cọc 500 triệu đã đóng. Chị Hương yêu cầu hoàn tiền, bồi thường do chậm bàn giao và chi phí thuê nhà 18 tháng.",
        "dispute": ["Tranh chấp hợp đồng mua bán căn hộ", "Chậm bàn giao bất động sản"],
        "law": ["430", "434", "351", "360"],
        "features": {
            "parties_info": ["chủ đầu tư bất động sản", "công ty"],
            "dispute_acts": [
                "không bàn giao đúng hạn",
                "từ chối hoàn cọc",
                "chậm tiến độ 18 tháng",
            ],
            "subject_matter": [
                "tiền cọc 500 triệu",
                "chi phí thuê nhà 18 tháng",
                "thiệt hại cơ hội",
            ],
            "fault_and_evidence": ["vi phạm hợp đồng có chủ ý"],
        },
    },
    {
        "name": ["Phan Văn Dũng", "Phan Thị Nga"],
        "fact": "Ông Phan Văn Dũng và bà Phan Thị Nga ly hôn sau 20 năm chung sống. Tài sản tranh chấp gồm: căn nhà tại Hà Nội trị giá 5 tỷ đồng, một mảnh đất thừa kế của bà Nga 1,5 tỷ và tiết kiệm chung 800 triệu. Ông Dũng yêu cầu chia đều tất cả tài sản. Bà Nga yêu cầu giữ lại tài sản thừa kế riêng và chia đều phần tài sản chung. Tranh chấp còn liên quan đến quyền nuôi con 10 tuổi.",
        "dispute": ["Tranh chấp tài sản khi ly hôn", "Tranh chấp tài sản riêng và chung"],
        "law": ["33", "43", "48", "81"],
        "features": {
            "parties_info": ["vợ chồng ly hôn", "cá nhân"],
            "dispute_acts": ["tranh chấp phân chia tài sản", "tranh chấp quyền nuôi con"],
            "subject_matter": ["nhà 5 tỷ", "đất thừa kế 1,5 tỷ", "tiết kiệm 800 triệu"],
            "fault_and_evidence": [],
        },
    },
    {
        "name": ["Mai Văn Tâm"],
        "fact": "Ông Mai Văn Tâm giao xe máy cho anh Hoàng Kim Sơn mượn trong 1 tuần. Anh Sơn cho người khác mượn tiếp mà không có sự đồng ý của ông Tâm. Người mượn xe gây tai nạn làm hỏng xe nặng, thiệt hại 45 triệu đồng. Ông Tâm yêu cầu anh Sơn bồi thường toàn bộ thiệt hại. Anh Sơn từ chối với lý do không trực tiếp gây tai nạn.",
        "dispute": ["Tranh chấp bồi thường thiệt hại tài sản", "Vi phạm hợp đồng mượn tài sản"],
        "law": ["494", "496", "584", "585"],
        "features": {
            "parties_info": ["người mượn tài sản", "cá nhân"],
            "dispute_acts": [
                "cho mượn tiếp tài sản không được phép",
                "không bảo quản tài sản",
                "từ chối bồi thường",
            ],
            "subject_matter": ["xe máy hư hỏng 45 triệu đồng"],
            "fault_and_evidence": ["sơ suất trong quản lý tài sản mượn"],
        },
    },
    {
        "name": ["Trịnh Thị Hà"],
        "fact": "Bà Trịnh Thị Hà sở hữu mảnh đất 200m² tại Bình Dương, có Giấy chứng nhận quyền sử dụng đất. Ông Nguyễn Hải xây tường rào lấn chiếm 20m² diện tích đất của bà mà không có thỏa thuận. Bà Hà nhiều lần yêu cầu tháo dỡ nhưng ông Hải tiếp tục xây thêm công trình trên phần đất lấn chiếm. Bà Hà khởi kiện yêu cầu phá dỡ công trình, hoàn trả đất và bồi thường thiệt hại.",
        "dispute": ["Tranh chấp lấn chiếm đất đai", "Tranh chấp quyền sử dụng đất"],
        "law": ["166", "169", "175"],
        "features": {
            "parties_info": ["người lấn chiếm đất", "hàng xóm"],
            "dispute_acts": [
                "xây tường lấn chiếm 20m²",
                "tiếp tục xây công trình trên đất chiếm",
                "không tuân thủ yêu cầu tháo dỡ",
            ],
            "subject_matter": ["20m² đất bị chiếm dụng", "thiệt hại quyền sử dụng đất"],
            "fault_and_evidence": ["cố ý lấn chiếm có hệ thống"],
        },
    },
    {
        "name": ["Bùi Văn Quang"],
        "fact": "Ông Bùi Văn Quang ký hợp đồng thuê cửa hàng với bà Lý Thị Lan, giá 15 triệu/tháng, thời hạn 3 năm từ 1/2021, đặt cọc 90 triệu. Sau 18 tháng, bà Lan đơn phương chấm dứt hợp đồng với lý do cần sử dụng lại mặt bằng, không thông báo trước theo hợp đồng và từ chối hoàn trả tiền cọc. Ông Quang yêu cầu tiếp tục thuê hoặc hoàn tiền cọc và bồi thường thiệt hại kinh doanh.",
        "dispute": ["Tranh chấp hợp đồng thuê tài sản", "Không hoàn trả tiền cọc"],
        "law": ["472", "478", "328"],
        "features": {
            "parties_info": ["bên cho thuê", "cá nhân"],
            "dispute_acts": [
                "đơn phương chấm dứt hợp đồng thuê",
                "giữ tiền cọc bất hợp pháp",
                "không thông báo trước",
            ],
            "subject_matter": ["tiền cọc 90 triệu", "thiệt hại kinh doanh 18 tháng còn lại"],
            "fault_and_evidence": ["cố ý vi phạm hợp đồng thuê"],
        },
    },
    {
        "name": ["Nguyễn Thị Phương"],
        "fact": "Bà Nguyễn Thị Phương nhận được di chúc của cha để lại toàn bộ tài sản gồm nhà đất trị giá 8 tỷ đồng. Hai người anh trai phản đối di chúc với lý do cha không có đủ năng lực hành vi tại thời điểm lập do mắc bệnh Alzheimer giai đoạn 2. Có bằng chứng y tế cho thấy bệnh nhân vẫn còn nhận thức. Tranh chấp thừa kế liên quan đến hiệu lực di chúc và phần di sản tối thiểu theo pháp luật.",
        "dispute": ["Tranh chấp thừa kế theo di chúc", "Tranh chấp năng lực lập di chúc"],
        "law": ["625", "627", "630", "644"],
        "features": {
            "parties_info": ["những người thừa kế", "anh em ruột"],
            "dispute_acts": ["phản đối tính hợp lệ của di chúc", "tranh chấp di sản"],
            "subject_matter": ["nhà đất 8 tỷ đồng", "quyền thừa kế hợp pháp"],
            "fault_and_evidence": [],
        },
    },
    {
        "name": ["Đỗ Thanh Tùng"],
        "fact": "Anh Đỗ Thanh Tùng bị tai nạn giao thông do xe tải của Công ty Logistics Miền Nam gây ra khi xe tải vượt đèn đỏ. Anh bị thương nặng, gãy xương chậu, nằm viện 3 tháng, chi phí y tế 320 triệu, mất thu nhập 12 tháng 180 triệu. Công ty chỉ đồng ý bồi thường 100 triệu qua bảo hiểm xe. Anh yêu cầu bồi thường toàn bộ thiệt hại thực tế theo Bộ luật Dân sự.",
        "dispute": ["Bồi thường thiệt hại ngoài hợp đồng", "Tai nạn giao thông gây thương tích"],
        "law": ["584", "589", "590"],
        "features": {
            "parties_info": ["người gây tai nạn", "công ty vận tải"],
            "dispute_acts": ["vượt đèn đỏ gây tai nạn", "từ chối bồi thường đủ"],
            "subject_matter": [
                "chi phí y tế 320 triệu",
                "mất thu nhập 180 triệu",
                "tổn thất tinh thần",
            ],
            "fault_and_evidence": ["sơ suất khi điều khiển phương tiện"],
        },
    },
    {
        "name": ["Cao Minh Nhật"],
        "fact": "Anh Cao Minh Nhật đặt cọc 200 triệu để mua căn nhà của ông Vũ Thế Anh theo thỏa thuận đặt cọc ký ngày 5/1/2023, thời hạn 60 ngày hoàn tất công chứng. Trong thời hạn, anh Nhật đã chuẩn bị đủ tiền và yêu cầu ký hợp đồng chuyển nhượng nhưng ông Vũ từ chối vì tìm được người mua khác với giá cao hơn. Anh Nhật yêu cầu hoàn trả cọc và bồi thường thêm một khoản tương đương tiền cọc.",
        "dispute": ["Tranh chấp đặt cọc mua bán nhà đất", "Vi phạm thỏa thuận đặt cọc"],
        "law": ["328", "430"],
        "features": {
            "parties_info": ["bên bán nhà", "cá nhân"],
            "dispute_acts": [
                "từ chối thực hiện giao dịch đã đặt cọc",
                "vi phạm thỏa thuận đặt cọc",
            ],
            "subject_matter": ["tiền cọc 200 triệu đồng", "thiệt hại cơ hội mua nhà"],
            "fault_and_evidence": ["cố ý vi phạm để bán cho người khác"],
        },
    },
]

BAO_HIEM_CASES = [
    {
        "name": ["Trương Văn Hải"],
        "fact": "Ông Trương Văn Hải đóng bảo hiểm xã hội bắt buộc liên tục 28 năm đến khi nghỉ hưu năm 2022. Cơ quan BHXH tỉnh tính lương hưu của ông dựa trên mức lương 5 năm cuối theo quy định cũ thay vì áp dụng quy định mới có lợi hơn. Mức hưu thực nhận thấp hơn khoảng 3 triệu/tháng so với tính đúng quy định. Ông Hải khiếu nại yêu cầu tính lại lương hưu.",
        "dispute": ["Tranh chấp tính lương hưu", "Vi phạm quy định bảo hiểm xã hội"],
        "law": ["56", "74", "89"],
        "features": {
            "parties_info": ["cơ quan BHXH", "cơ quan hành chính"],
            "dispute_acts": ["tính lương hưu sai quy định", "áp dụng sai văn bản pháp luật"],
            "subject_matter": ["thiệt hại 3 triệu/tháng lương hưu", "thiệt hại lũy kế"],
            "fault_and_evidence": ["nhầm lẫn trong áp dụng pháp luật"],
        },
    },
    {
        "name": ["Nguyễn Thị Dung"],
        "fact": "Chị Nguyễn Thị Dung sinh con tháng 5/2023, có đủ 6 tháng đóng BHXH trong 12 tháng trước sinh. Công ty nơi chị làm việc không khai báo đầy đủ quá trình đóng BHXH của chị, làm chị bị cơ quan BHXH từ chối chi trả chế độ thai sản. Chị Dung phải chứng minh lại quá trình đóng qua hồ sơ lương và sổ BHXH. Tranh chấp kéo dài 8 tháng, ảnh hưởng đến quyền lợi thai sản.",
        "dispute": ["Từ chối hưởng chế độ thai sản", "Báo cáo BHXH sai"],
        "law": ["31", "32", "34"],
        "features": {
            "parties_info": ["người sử dụng lao động", "cơ quan BHXH"],
            "dispute_acts": ["không khai báo đóng BHXH đầy đủ", "gây ảnh hưởng quyền lợi thai sản"],
            "subject_matter": ["mất chế độ thai sản 6 tháng", "thiệt hại tài chính"],
            "fault_and_evidence": ["sơ suất trong quản lý BHXH"],
        },
    },
    {
        "name": ["Lý Văn Phúc"],
        "fact": "Anh Lý Văn Phúc làm công nhân xây dựng 15 năm, bị tai nạn lao động nghiêm trọng năm 2021 dẫn đến mất 65% sức lao động. Hội đồng giám định y khoa xác nhận tỷ lệ thương tật. Cơ quan BHXH tính sai tỷ lệ hưởng, thiếu trợ cấp hàng tháng 1,8 triệu đồng. Anh Phúc khiếu nại và yêu cầu tính lại từ thời điểm tai nạn, thu hồi số tiền đã thiếu.",
        "dispute": ["Tranh chấp chế độ tai nạn lao động", "Tính sai tỷ lệ trợ cấp"],
        "law": ["43", "48", "49"],
        "features": {
            "parties_info": ["cơ quan BHXH", "cơ quan hành chính"],
            "dispute_acts": ["tính sai tỷ lệ hưởng BHXH tai nạn", "thiếu trợ cấp hàng tháng"],
            "subject_matter": ["thiếu 1,8 triệu/tháng", "thiệt hại lũy kế nhiều năm"],
            "fault_and_evidence": ["nhầm lẫn tính toán"],
        },
    },
    {
        "name": ["Phan Thị Bảo"],
        "fact": "Bà Phan Thị Bảo đóng BHXH tự nguyện liên tục 10 năm. Khi đủ tuổi nghỉ hưu và đủ 20 năm đóng BHXH, bà xin hưởng lương hưu nhưng bị từ chối vì cơ quan BHXH không tính giai đoạn đóng BHXH tự nguyện vào thời gian hưởng lương hưu. Bà Bảo phản đối vì theo Luật BHXH 2014, BHXH tự nguyện được tích lũy cùng BHXH bắt buộc để hưởng lương hưu.",
        "dispute": ["Tranh chấp điều kiện hưởng lương hưu", "Vi phạm quy định BHXH tự nguyện"],
        "law": ["73", "74", "87", "88"],
        "features": {
            "parties_info": ["cơ quan BHXH", "cơ quan hành chính"],
            "dispute_acts": [
                "từ chối điều kiện hưởng lương hưu",
                "không tính BHXH tự nguyện vào thời gian đóng",
            ],
            "subject_matter": ["quyền hưởng lương hưu", "thiệt hại quyền lợi hưu trí"],
            "fault_and_evidence": ["sai sót trong áp dụng pháp luật BHXH"],
        },
    },
    {
        "name": ["Vương Đình Hoà"],
        "fact": "Ông Vương Đình Hoà làm việc tại doanh nghiệp tư nhân 12 năm. Khi doanh nghiệp phá sản, chủ doanh nghiệp không đóng đủ BHXH cho ông trong 3 năm cuối dù đã khấu trừ từ lương. Số tiền BHXH bị chiếm dụng khoảng 85 triệu đồng. Khi ông đề nghị rút BHXH một lần hoặc hưởng lương hưu, thời gian đóng bị tính thiếu 3 năm. Ông yêu cầu cơ quan BHXH và thanh lý tài sản doanh nghiệp phá sản giải quyết bổ sung.",
        "dispute": ["Chiếm dụng tiền đóng BHXH của người lao động", "Không đóng BHXH đủ"],
        "law": ["17", "89", "216"],
        "features": {
            "parties_info": ["người sử dụng lao động phá sản", "doanh nghiệp tư nhân"],
            "dispute_acts": [
                "khấu trừ BHXH từ lương nhưng không đóng",
                "chiếm dụng tiền BHXH",
                "gian lận báo cáo BHXH",
            ],
            "subject_matter": ["85 triệu tiền BHXH bị chiếm dụng", "3 năm đóng BHXH bị tính thiếu"],
            "fault_and_evidence": ["cố ý chiếm dụng tiền đóng BHXH"],
        },
    },
    {
        "name": ["Hà Thị Tuyết"],
        "fact": "Chị Hà Thị Tuyết bị mất việc làm tháng 8/2023 do công ty thu hẹp sản xuất. Chị đủ điều kiện hưởng trợ cấp thất nghiệp (đóng đủ 12 tháng bảo hiểm thất nghiệp). Tuy nhiên, trung tâm dịch vụ việc làm từ chối hồ sơ với lý do thiếu giấy tờ không cần thiết theo quy định. Sau 3 lần bổ sung hồ sơ, chị vẫn chưa nhận được tiền trợ cấp sau 4 tháng. Chị khởi kiện yêu cầu thanh toán trợ cấp thất nghiệp và lãi chậm trả.",
        "dispute": ["Từ chối hưởng trợ cấp thất nghiệp", "Trì hoãn giải quyết hồ sơ BHXH"],
        "law": ["49", "50", "52"],
        "features": {
            "parties_info": ["trung tâm dịch vụ việc làm", "cơ quan hành chính"],
            "dispute_acts": ["từ chối hồ sơ không đúng quy định", "trì hoãn giải quyết 4 tháng"],
            "subject_matter": ["mất trợ cấp thất nghiệp 4 tháng", "thiệt hại tài chính"],
            "fault_and_evidence": ["hành chính sai quy trình"],
        },
    },
    {
        "name": ["Đặng Minh Quân"],
        "fact": "Anh Đặng Minh Quân mắc bệnh nghề nghiệp (bụi phổi) sau 20 năm làm việc trong môi trường khai thác than. Hội đồng giám định y khoa xác định thương tật 45%. Anh được hưởng trợ cấp bệnh nghề nghiệp một lần nhưng số tiền tính không đúng do cơ quan BHXH dùng sai mức lương bình quân. Anh Quân yêu cầu tính lại dựa trên mức lương thực tế và nhận phần chênh lệch bị thiếu.",
        "dispute": ["Tranh chấp chế độ bệnh nghề nghiệp", "Tính sai trợ cấp bệnh nghề nghiệp"],
        "law": ["43", "44", "45", "46"],
        "features": {
            "parties_info": ["cơ quan BHXH", "cơ quan hành chính"],
            "dispute_acts": [
                "tính sai mức lương bình quân",
                "xác định sai trợ cấp bệnh nghề nghiệp",
            ],
            "subject_matter": ["thiếu trợ cấp bệnh nghề nghiệp", "thiệt hại quyền lợi bảo hiểm"],
            "fault_and_evidence": ["sai sót trong tính toán"],
        },
    },
]


def build_cases_json():
    all_cases = []
    case_id = 0
    for domain, cases in [
        ("lao_dong", LAO_DONG_CASES),
        ("dan_su", DAN_SU_CASES),
        ("bao_hiem", BAO_HIEM_CASES),
    ]:
        for case in cases:
            entry = {
                "id": case_id,
                "name": case["name"],
                "fact": case["fact"],
                "dispute": case["dispute"],
                "law": case["law"],
                "laws": case["law"],
                "domain": domain,
                "features": case["features"],
            }
            all_cases.append(entry)
            case_id += 1
    return all_cases


def main():
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "cases_with_feature.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cases = build_cases_json()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    domain_counts = {}
    for c in cases:
        d = c["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    logger.info(f"Generated {len(cases)} sample cases -> {output_path}")
    for domain, count in domain_counts.items():
        logger.info(f"  {domain}: {count} cases")


if __name__ == "__main__":
    main()
