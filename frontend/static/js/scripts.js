$(document).ready(function() {
    $('#process-btn').click(function() {
        const inputText = $('#input-text').val();

        $.ajax({
            url: '/process',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: inputText }),
            success: function(response) {
                if(response.status === 'success') {
                    $('#result-container').html(`
                        <h3>检测结果：</h3>
                        <p>${response.result}</p>
                    `);
                }
            },
            error: function(xhr) {
                console.error('请求失败:', xhr.responseText);
            }
        });
    });
});