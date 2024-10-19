/* User Key */
CREATE TABLE `tb_user_key` (
    user_id INT NOT NULL AUTO_INCREMENT,
    uk_email VARCHAR(55) NOT NULL,
    uk_password VARCHAR(100) NOT NULL,
    uk_permission BOOL NOT NULL,
    PRIMARY KEY (user_id)
);

/* User Information */
CREATE TABLE `tb_user_information` (
    ui_id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    ui_name VARCHAR(35) NULL,
    ui_birth_date DATE NULL,
    ui_sex BOOL NULL,
    ui_bank_num VARCHAR(100) NULL,
    ui_caution INT NULL,
    ui_phone_number VARCHAR(9) NOT NULL UNIQUE,
    PRIMARY KEY (ui_id),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* User Finance */
CREATE TABLE `tb_user_finance` (
    uf_id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    uf_capital BIGINT NULL,
    uf_loan INT NULL,
    uf_installment_saving INT NULL,
    uf_deposit INT NULL,
    uf_target_budget INT NULL,
    PRIMARY KEY (uf_id),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* Chat Bot */
CREATE TABLE tb_chat_bot (
    cb_id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    cb_text TEXT,
    cb_query TEXT,
    cb_division BOOL NOT NULL,
    PRIMARY KEY (cb_id),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* Finance Date */
CREATE TABLE `tb_finance_date` (
    fd_date DATE NOT NULL,
    PRIMARY KEY (fd_date)
);

/* Korea Economic Indicator */
CREATE TABLE tb_korea_economic_indicator (
    kei_id INT NOT NULL AUTO_INCREMENT,
    fd_date DATE NOT NULL,
    kei_gdp FLOAT NOT NULL,
    kei_m2_end FLOAT NOT NULL,
    kei_m2_avg FLOAT NOT NULL,
    kei_fed_rate FLOAT NOT NULL,
    kei_ppi FLOAT NOT NULL,
    kei_ipi FLOAT NOT NULL,
    kei_cpi FLOAT NOT NULL,
    kei_imp FLOAT NOT NULL,
    kei_exp FLOAT NOT NULL,
    kei_cs FLOAT NOT NULL,
    kei_bsi FLOAT NOT NULL,
    PRIMARY KEY (kei_id),
    FOREIGN KEY (fd_date) REFERENCES tb_finance_date(fd_date)
);

/* US Economic Indicator */
CREATE TABLE tb_us_economic_indicator (
    uei_id INT NOT NULL AUTO_INCREMENT,
    fd_date DATE NOT NULL,
    uei_gdp FLOAT NOT NULL,
    uei_fed_rate FLOAT NOT NULL,
    uei_ipi FLOAT NOT NULL,
    uei_ppi FLOAT NOT NULL,
    uei_cpi FLOAT NOT NULL,
    uei_cpi_m FLOAT NOT NULL,
    uei_trade FLOAT NOT NULL,
    uei_cb_cc FLOAT NOT NULL,
    uei_ps_m FLOAT NOT NULL,
    uei_rs_m FLOAT NOT NULL,
    uei_umich_cs FLOAT NOT NULL,
    PRIMARY KEY (uei_id),
    FOREIGN KEY (fd_date) REFERENCES tb_finance_date(fd_date)
);

/* Main Economic Index */
CREATE TABLE tb_main_economic_index (
    mei_id INT NOT NULL AUTO_INCREMENT,
    fd_date DATE NOT NULL,
    mei_nasdaq FLOAT NOT NULL,
    mei_sp500 FLOAT NOT NULL,
    mei_dow FLOAT NOT NULL,
    mei_kospi FLOAT NOT NULL,
    mei_gold FLOAT NOT NULL,
    mei_oil FLOAT NOT NULL,
    mei_ex_rate FLOAT NOT NULL,
    PRIMARY KEY (mei_id),
    FOREIGN KEY (fd_date) REFERENCES tb_finance_date(fd_date)
);

/* Stock */
CREATE TABLE tb_stock (
    sc_id INT NOT NULL AUTO_INCREMENT,
    fd_date DATE NOT NULL,
    sc_ss_stock FLOAT NOT NULL,
    sc_ss_per FLOAT NOT NULL,
    sc_ss_pbr FLOAT NOT NULL,
    sc_ss_roe FLOAT NOT NULL,
    sc_ss_mc FLOAT NOT NULL,
    sc_ap_stock FLOAT NOT NULL,
    sc_ap_per FLOAT NOT NULL,
    sc_ap_pbr FLOAT NOT NULL,
    sc_ap_roe FLOAT NOT NULL,
    sc_ap_mc FLOAT NOT NULL,
    sc_coin FLOAT NOT NULL,
    PRIMARY KEY (sc_id),
    FOREIGN KEY (fd_date) REFERENCES tb_finance_date(fd_date)
);

/* stock predict */
CREATE TABLE tb_stock_predict (
    sp_id INT NOT NULL AUTO_INCREMENT,
    sp_date DATE NOT NULL,
    sp_ss_predict FLOAT NOT NULL,
    sp_ap_predict FLOAT NOT NULL,
    sp_bit_predict FLOAT NOT NULL,
    PRIMARY KEY (sp_id)
);

/* Shares Held */
CREATE TABLE tb_shares_held (
    sh_id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    sh_date DATE NOT NULL,
    sh_ss_count INT,
    sh_ap_count INT,
    sh_bit_count float,
    PRIMARY KEY (sh_id),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* Received Paid */
CREATE TABLE tb_received_paid (
    rp_id INT NOT NULL AUTO_INCREMENT,
    user_id INT NOT NULL,
    rp_date DATE NOT NULL,
    rp_detail VARCHAR(100) NOT NULL,
    rp_amount DOUBLE NOT NULL,
    rp_hold BOOL NOT NULL,
    rp_part BOOL NOT NULL,
    PRIMARY KEY (rp_id),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* Finance Memo */
CREATE TABLE tb_finance_memo (
    fm_date DATE NOT NULL,
    user_id INT NOT NULL,
    fm_memo TEXT,
    PRIMARY KEY (fm_date),
    FOREIGN KEY (user_id) REFERENCES tb_user_key(user_id) ON DELETE CASCADE
);

/* News */
CREATE TABLE tb_news (
    news_id INT NOT NULL AUTO_INCREMENT,
    news_title VARCHAR(100) NOT NULL,
    news_simple_text LONGTEXT NOT NULL,
    news_link TEXT NOT NULL,
    news_classification BOOL NOT NULL,
    PRIMARY KEY (news_id)
);

CREATE TABLE tb_news_chat_room (
    ncr_id INT NOT NULL AUTO_INCREMENT,
    ncr_chat_message TEXT,
    ncr_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ncr_id)
);

/* 인코딩 변경 */
ALTER TABLE tb_user_key 
MODIFY uk_email VARCHAR(55) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
MODIFY uk_password VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL;

ALTER TABLE tb_user_information 
MODIFY ui_name VARCHAR(35) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
MODIFY ui_bank_num VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
MODIFY ui_phone_number VARCHAR(9) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL;

ALTER TABLE tb_received_paid
MODIFY rp_detail VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL;

ALTER TABLE tb_chat_bot 
MODIFY cb_text TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
MODIFY cb_query TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

ALTER TABLE tb_news  
MODIFY news_title VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
MODIFY news_simple_text LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
MODIFY news_link TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL;

ALTER TABLE tb_news_chat_room 
MODIFY ncr_chat_message TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

ALTER TABLE tb_finance_memo 
MODIFY fm_memo TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

/* 기타 설정 */
SET GLOBAL innodb_lock_wait_timeout = 1000;

SHOW TABLES;
