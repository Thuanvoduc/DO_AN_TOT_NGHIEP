const ITEMS_PER_PAGE = 15; // Số bản ghi mỗi trang
let currentPage = 1;

// dữ liệu mẫu để test
const records = [
  { time: "2025-10-10", status: "Len-xe" },
  { time: "2025-10-10", status: "Xuong-xe" },
  { time: "2025-10-10", status: "Len-xe" },
  { time: "2025-10-10", status: "Xuong-xe" },

  { time: "2025-10-11", status: "Len-xe" },
  { time: "2025-10-11", status: "Xuong-xe" },
  { time: "2025-10-11", status: "Len-xe" },
  { time: "2025-10-11", status: "Xuong-xe" },

  { time: "2025-10-12", status: "Len-xe" },
  { time: "2025-10-12", status: "Xuong-xe" },
  { time: "2025-10-12", status: "Len-xe" },
  { time: "2025-10-12", status: "Xuong-xe" },

  { time: "2025-10-13", status: "Len-xe" },
  { time: "2025-10-13", status: "Xuong-xe" },
  { time: "2025-10-13", status: "Len-xe" },
  { time: "2025-10-13", status: "Xuong-xe" },

  { time: "2025-10-14", status: "Len-xe" },
  { time: "2025-10-14", status: "Xuong-xe" },
  { time: "2025-10-14", status: "Len-xe" },
  { time: "2025-10-14", status: "Xuong-xe" },

  { time: "2025-10-15", status: "Len-xe" },
  { time: "2025-10-15", status: "Xuong-xe" },
  { time: "2025-10-15", status: "Len-xe" },
  { time: "2025-10-15", status: "Xuong-xe" },

  { time: "2025-10-16", status: "Len-xe" },
  { time: "2025-10-16", status: "Xuong-xe" },
  { time: "2025-10-16", status: "Len-xe" },
  { time: "2025-10-16", status: "Xuong-xe" },

  { time: "2025-10-17", status: "Len-xe" },
  { time: "2025-10-17", status: "Xuong-xe" },
  { time: "2025-10-17", status: "Len-xe" },
  { time: "2025-10-17", status: "Xuong-xe" },

  { time: "2025-10-18", status: "Len-xe" },
  { time: "2025-10-18", status: "Xuong-xe" },
  { time: "2025-10-18", status: "Len-xe" },
  { time: "2025-10-18", status: "Xuong-xe" }
];

// Hiện dữ liệu quẹt thẻ trong 5 ngày gần nhất
function renderRFIDBarChart(records) {
  // 🔹 Nhóm dữ liệu theo ngày
  const groupedData = {};
  $.each(records, function (_, record) {
    const date = record.time.split("T")[0]; // YYYY-MM-DD
    if (!groupedData[date]) {
      groupedData[date] = { up: 0, down: 0 };
    }
    if (record.status === "Len-xe") groupedData[date].up++;
    if (record.status === "Xuong-xe") groupedData[date].down++;
  });

  // 🔹 Lấy danh sách ngày (đã sắp xếp)
  const sortedDates = Object.keys(groupedData).sort();

  // 🔹 Giữ lại tối đa 5 ngày gần nhất
  const recentDates = sortedDates.slice(-5);

  // 🔹 Chuẩn bị dữ liệu cho chart
  const upData = recentDates.map(date => groupedData[date].up);
  const downData = recentDates.map(date => groupedData[date].down);

  // 🔹 Xóa chart cũ (nếu có) để tránh đè lên nhau
  if (window.rfidChart) {
    window.rfidChart.destroy();
  }

  // 🔹 Khởi tạo chart mới
  const ctx = $('#rfidBarChart')[0].getContext('2d');
  window.rfidChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: recentDates,
      datasets: [
        {
          label: 'Lên xe',
          data: upData,
          backgroundColor: 'rgba(92, 75, 153, 0.8)',
          borderRadius: 6
        },
        {
          label: 'Xuống xe',
          data: downData,
          backgroundColor: 'rgba(166, 146, 226, 0.8)',
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true, // ⚠️ Giữ nguyên tỉ lệ, không bể layout
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#333' }
        },
        title: {
          display: true,
          text: 'Biểu đồ số lần lên - xuống xe (5 ngày gần nhất)',
          color: '#5c4b99',
          font: { size: 16, weight: 'bold' }
        }
      },
      scales: {
        x: {
          ticks: { color: '#333' },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#333', stepSize: 1 },
          grid: { color: '#eee' }
        }
      }
    }
  });
}

// Hiện dữ liệu số ngày đã đi học và chưa đi học trong tuần 
function renderWeeklyAttendanceChart(records) {
  const uniqueDays = new Set(records.map(r => r.time.split("T")[0]));
  const daysPresent = uniqueDays.size;
  const totalDays = 7; // 1 tuần = 7 ngày
  const daysAbsent = totalDays - daysPresent;

  const ctx = $('#weeklyAttendanceChart')[0].getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Đi học', 'Vắng'],
      datasets: [{
        data: [daysPresent, daysAbsent],
        backgroundColor: ['rgba(92, 75, 153, 0.8)', 'rgba(200, 200, 200, 0.6)'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#333' }
        },
        title: {
          display: true,
          text: 'Số ngày đã đi học trong một tuần',
          color: '#5c4b99',
          font: { size: 16, weight: 'bold' }
        }
      }
    }
  });
}

// Hiện dữ liệu Tổng quan điểm danh 
function updateAttendanceSummary(records) {
  const totalSwipes = records.length;

  const uniqueDays = new Set(records.map(r => r.time.split("T")[0]));
  const daysPresent = uniqueDays.size;

  const totalDays = 8; // giả sử trong tháng có 8 ngày học (tuỳ dữ liệu thật)
  const daysAbsent = totalDays - daysPresent;

  const lastSwipe = records.length ? new Date(records[records.length - 1].time).toLocaleString() : "Không có";

  $("#totalSwipes").text(totalSwipes);
  $("#daysPresent").text(daysPresent);
  $("#daysAbsent").text(daysAbsent);
  $("#lastSwipe").text(lastSwipe);
}

function renderHistory(records) {
  const tbody = $("#historyList");
  tbody.empty();

  if (!records || records.length === 0) {
    tbody.append(`<tr><td colspan="2" class="text-center">Không có dữ liệu</td></tr>`);
    $("#pagination").empty();
    return;
  }

  // 🔹 Sắp xếp từ mới → cũ
  const sorted = [...records].sort((a, b) => new Date(b.time) - new Date(a.time));

  // 🔹 Tính số trang
  const totalPages = Math.ceil(sorted.length / ITEMS_PER_PAGE);

  // 🔹 Giới hạn chỉ mục trang hợp lệ
  if (currentPage > totalPages) currentPage = totalPages;

  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const pageData = sorted.slice(startIndex, endIndex);

  // 🔹 Render dữ liệu trang hiện tại
  pageData.forEach(r => {
    const statusText = r.status === "Len-xe" ? "🚍 Lên xe" : "🏫 Xuống xe";
    const row = `
      <tr>
        <td>${r.time}</td>
        <td>${statusText}</td>
      </tr>
    `;
    tbody.append(row);
  });

  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  const pagination = $("#pagination");
  pagination.empty();

  // 🔹 Nếu chỉ có 1 trang thì ẩn luôn
  if (totalPages <= 1) return;

  // Nút Previous
  const prevDisabled = currentPage === 1 ? "disabled" : "";
  pagination.append(`<button ${prevDisabled} id="prevPage">⬅</button>`);

  // Nút số trang
  for (let i = 1; i <= totalPages; i++) {
    const active = i === currentPage ? "active" : "";
    pagination.append(`<button class="page-btn ${active}" data-page="${i}">${i}</button>`);
  }

  // Nút Next
  const nextDisabled = currentPage === totalPages ? "disabled" : "";
  pagination.append(`<button ${nextDisabled} id="nextPage">➡</button>`);
}

$(document).ready(function () {
  // ✅ Xử lý khi click Sidebar
  $(".nav-link").click(function (e) {
    e.preventDefault();

    // 1️⃣ Bỏ active ở tất cả, thêm active cho cái đang chọn
    $(".nav-link").removeClass("active");
    $(this).addClass("active");

    // 2️⃣ Lấy section tương ứng
    const section = $(this).data("section");

    // 3️⃣ Ẩn tất cả section, chỉ hiện section đang chọn
    $(".content-section").addClass("d-none");
    $(`#section-${section}`).removeClass("d-none");

    // 4️⃣ Nếu click vào “Lịch sử điểm danh” thì render dữ liệu
    if (section === "history") {
      renderHistory(records);
    }
  });

  $("#logoutBtn").click(function () {
    alert("Đăng xuất thành công!");
  });

  // 🔹 Sự kiện click phân trang
  $(document).on("click", ".page-btn", function () {
    currentPage = parseInt($(this).data("page"));
    renderHistory(records);
  });

  $(document).on("click", "#prevPage", function () {
    if (currentPage > 1) {
      currentPage--;
      renderHistory(records);
    }
  });

  $(document).on("click", "#nextPage", function () {
    const totalPages = Math.ceil(records.length / ITEMS_PER_PAGE);
    if (currentPage < totalPages) {
      currentPage++;
      renderHistory(records);
    }
  });

  // Gọi hàm vẽ biểu đồ 
  renderRFIDBarChart(records);

  // Gọi hàm vẽ số ngày đi học và vắng trong Tuần 
  renderWeeklyAttendanceChart(records);

  // Tổng quan điểm danh
  updateAttendanceSummary(records)
});
