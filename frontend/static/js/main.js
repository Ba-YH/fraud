document.addEventListener('DOMContentLoaded', function() {
    // 主题切换
    const themeSwitch = document.querySelector('.theme-switch');
    const body = document.body;

    // 从本地存储加载主题
    const currentTheme = localStorage.getItem('theme') || 'light';
    body.setAttribute('data-theme', currentTheme);

    themeSwitch.addEventListener('click', () => {
        const theme = body.getAttribute('data-theme');
        const newTheme = theme === 'light' ? 'dark' : 'light';
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // 移动端侧边栏切换
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });

    // 点击主内容区域时关闭侧边栏
    document.querySelector('.main-content').addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('active')) {
            sidebar.classList.remove('active');
        }
    });
});